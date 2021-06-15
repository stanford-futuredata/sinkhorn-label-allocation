#
# Wrappers for model evaluation
#

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from modules import Classifier
from typing import Generator, NamedTuple, Optional, Union
from utils import expand_generator


class Evaluator(object):
    class Result(NamedTuple):
        accuracy: float
        log_loss: float

    def evaluate(self, *args, **kwargs):
        return NotImplemented


class ModelEvaluator(Evaluator):
    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int, mixed_precision: bool = True):
        self.dataset = dataset
        self.mixed_precision = mixed_precision
        self.loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    @property
    def num_batches(self):
        return len(self.loader)

    def evaluate(self, model: Classifier, device: Optional[Union[torch.device, str]] = None) -> Evaluator.Result:
        return expand_generator(self.evaluate_iter(model, device), return_only=True)

    def evaluate_iter(
            self,
            model: Classifier,
            device: Optional[Union[torch.device, str]] = None) -> Generator[dict, None, Evaluator.Result]:
        with model.as_eval(), torch.no_grad(), torch.cuda.amp.autocast(enabled=self.mixed_precision):
            mean_accuracy = 0.
            mean_log_loss = 0.
            for i, (x, y) in enumerate(self.loader):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                correct = torch.sum(logits.argmax(-1) == y).item()
                log_loss = F.cross_entropy(logits, y, reduction='sum').item()
                mean_accuracy += correct / len(self.dataset)
                mean_log_loss += log_loss / len(self.dataset)
                yield dict(batch=i)
            return self.Result(accuracy=mean_accuracy, log_loss=mean_log_loss)
