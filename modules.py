import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
import copy
import tqdm


#
# Parameter Averaging
#

class EMA(object):
    def __init__(self, model: nn.Module, alpha: float):
        self.model = model
        self.avg_model = copy.deepcopy(model)
        self.avg_model.float()
        self.alpha = alpha
        self.num_steps = 0

    def step(self):
        # update parameters
        for p, p_avg in zip(self.model.parameters(), self.avg_model.parameters()):
            p_avg.data.mul_(1 - self.alpha).add_(p.float(), alpha=self.alpha)

        # update buffers
        for p, p_avg in zip(self.model.buffers(), self.avg_model.buffers()):
            p_avg.data.copy_(p)

        self.num_steps += 1
        return self

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.avg_model.state_dict())
        return self


#
# Classifier base class
#

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / k))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def half(self):
        for module in self.children():
            if not isinstance(module, nn.BatchNorm2d):
                module.half()
        return self

    def update_batch_norm_stats(self, batches, momentum=None, device=None, print_progress=False):
        if not _check_bn(self):
            return
        was_training = self.training
        self.train()
        momenta = {}
        self.apply(_reset_bn)
        self.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        progress_hook = tqdm.tqdm if print_progress else lambda x: x
        with torch.no_grad():
            for input in progress_hook(batches):
                if isinstance(input, (list, tuple)):
                    input = input[0]
                b = input.size(0)

                momentum = b / float(n + b) if momentum is None else momentum
                for module in momenta.keys():
                    module.momentum = momentum

                if device is not None:
                    input = input.to(device)

                self(input)
                n += b

        self.apply(lambda module: _set_momenta(module, momenta))
        self.train(was_training)

    @contextmanager
    def as_train(self):
        mode = self.training
        self.train()
        yield
        self.train(mode)

    @contextmanager
    def as_eval(self):
        mode = self.training
        self.eval()
        yield
        self.train(mode)

    # from https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py
    @contextmanager
    def disable_tracking_bn_stats(self):
        def switch_attr(m):
            if hasattr(m, 'track_running_stats'):
                m.track_running_stats ^= True

        self.apply(switch_attr)
        yield
        self.apply(switch_attr)


def _check_bn_apply(module, flag):
    if issubclass(module.__class__, nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


#
# Simple convnet
#

class ConvPoolBN(nn.Module):
    def __init__(self, c_in, c_out, nonlin=nn.ReLU(), stride=1, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.pool = nn.MaxPool2d(pool_size)
        self.bn = nn.BatchNorm2d(c_out)
        self.nonlin = nonlin

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = self.bn(out)
        out = self.nonlin(out)
        return out


class BasicConvNet(Classifier):
    def __init__(self, num_classes, channels=(32, 64, 128, 256), nonlin=nn.ReLU()):
        super().__init__(num_classes)
        channels = [3, *channels]
        self.blocks = nn.Sequential(*[
            ConvPoolBN(channels[i], channels[i+1], nonlin=nonlin) for i in range(len(channels) - 1)])
        self.fc = nn.Linear(channels[-1], num_classes)
        self.initialize_parameters()

    def forward(self, x):
        out = self.blocks(x)
        out = F.max_pool2d(out, 2)
        out = out.flatten(1)
        out = self.fc(out)
        return out


#
# WideResNet
#

class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, nonlin, stride=1, activate_before_residual=False, bn_momentum=1e-3):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.nonlin = nonlin
        self.activate_before_residual = activate_before_residual
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in, momentum=bn_momentum)
        self.bn2 = nn.BatchNorm2d(c_out, momentum=bn_momentum)
        if c_in != c_out:
            self.conv_shortcut = nn.Conv2d(c_in, c_out, 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        out = self.nonlin(self.bn1(x))
        if self.activate_before_residual:
            x = out
        if self.c_in != self.c_out:
            x = self.conv_shortcut(x)
        out = self.conv1(out)
        out = self.nonlin(self.bn2(out))
        out = self.conv2(out)
        return x + out


# adapted from https://github.com/YU1ut/MixMatch-pytorch/blob/master/models/wideresnet.py
class WideResNet(Classifier):
    def __init__(self, num_classes, channels=32, block_depth=4, nonlin=nn.LeakyReLU(0.1), bn_momentum=1e-3, output_bias=True):
        super().__init__(num_classes)
        self.channels = channels
        self.block_depth = block_depth
        self.nonlin = nonlin

        self.conv = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.block1 = nn.Sequential(
            ResidualBlock(16, channels, nonlin, stride=1, activate_before_residual=True, bn_momentum=bn_momentum),
            *[ResidualBlock(channels, channels, nonlin, bn_momentum=bn_momentum) for _ in range(block_depth - 1)])

        self.block2 = nn.Sequential(
            ResidualBlock(channels, channels*2, nonlin, stride=2, bn_momentum=bn_momentum),
            *[ResidualBlock(channels*2, channels*2, nonlin, bn_momentum=bn_momentum) for _ in range(block_depth - 1)])

        self.block3 = nn.Sequential(
            ResidualBlock(channels*2, channels*4, nonlin, stride=2, bn_momentum=bn_momentum),
            *[ResidualBlock(channels*4, channels*4, nonlin, bn_momentum=bn_momentum) for _ in range(block_depth - 1)])

        self.bn = nn.BatchNorm2d(channels*4, momentum=bn_momentum)
        self.fc = nn.Linear(channels*4, num_classes, bias=output_bias)
        self.initialize_parameters()

    def forward(self, x, autocast=False):
        with torch.cuda.amp.autocast(enabled=autocast):
            out = self.conv(x)
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
            out = self.nonlin(self.bn(out))
            out = out.view(out.shape[0], out.shape[1], -1).mean(-1)
            out = self.fc(out)
        return out
