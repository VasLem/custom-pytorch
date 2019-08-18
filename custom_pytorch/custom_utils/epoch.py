"""Have ripped some part of code from segmentation_models_pytorch package and customized it
accordingly to make it more generic. TrainEpoch and ValidEpoch are classes to be used in the
according stages
"""

import sys

import torch
from torchnet.meter import AverageValueMeter
from tqdm import tqdm as tqdm


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, inp_index, gt_index, _logs=None):
        """
        :param dataloader: the pytorch dataloader
        :type dataloader: torch.utils.data.DataLoader
        :param inp_index: the index or key of the input inside each batch item
        :type inp_index: int|str
        :param gt_index: the index or key of the input inside each batch item
        :type gt_index: int|str
        :param _logs: if provided, a dictionary where logs are accumulated,
        cleared before iteration, defaults to None
        :type _logs: dict|None
        :return: the loss and metrics logs
        :rtype: dict
        """

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter()
                          for metric in self.metrics}
        if _logs is not None:
            _logs.clear()
        with tqdm(dataloader, desc=self.stage_name,
                  file=sys.stdout, disable=not self.verbose) as iterator:
            for item in iterator:
                x = item[inp_index]
                y = item[gt_index]
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y, logs=_logs)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
                try:
                    logs.update({'LR': self.optimizer.param_groups[0]['lr']})
                except AttributeError:
                    pass
                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, logs=None):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)

        loss = self.loss(prediction, y, logs=logs)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, logs=None):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y, logs=logs)
        return loss, prediction
