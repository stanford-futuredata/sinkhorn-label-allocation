import torch
import pandas as pd


class Logger(object):
    def write(self, ):
        pass


class TableLogger(Logger):
    def __init__(self):
        super().__init__()
        self.log = [{}]

    def __len__(self):
        return len(self.log) - 1

    def step(self):
        self.log.append(dict())

    def write(self, **kwargs):
        record = self.log[-1]
        for k, v in kwargs.items():
            record[k] = self._normalize(v)

    def append(self, **kwargs):
        record = self.log[-1]
        for k, v in kwargs.items():
            v = self._normalize(v)
            if k in record:
                record[k].append(v)
            else:
                record[k] = [v]

    @staticmethod
    def _normalize(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            x = x.item() if x.dim() == 0 else x.numpy()
        return x

    def to_dataframe(self):
        log = self.log
        if len(log[-1]) == 0:
            log = log[:-1]
        return pd.DataFrame.from_records(log)
