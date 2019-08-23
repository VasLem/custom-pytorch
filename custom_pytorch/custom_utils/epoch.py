"""Have ripped some part of code from segmentation_models_pytorch package and customized it
accordingly to make it more generic. TrainEpoch and ValidEpoch are classes to be used in the
according stages
"""

import sys
import numpy as np
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
        self.epoch = 0
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        if self.stage_name != 'test':
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

    def run(self, dataloader, inp_index, gt_index, _logs=None, _tta=False,
            _tta_strategy='mean', _tta_inv_transforms=None):
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
        :param _tta: if provided, it will be assumed that the input samples per batch are of
            (T, C, H, W) size, where T is the number of the test time augmentation samples.
            defaults to False.
        :type _tta: bool
        :param _tta_strategy: the strategy to follow for the predictions joining,
            available choices are:
                - mean
                - max
            Defaults to mean.
        :type _tta_strategy: str
        :param _tta_inv_transforms: the inverse tranforms to apply to predictions
        :type _tta_inv_transforms: callable
        :return: the loss and metrics logs
        :rtype: dict
        """
        self.epoch += 1
        self.on_epoch_start()

        logs = {}
        if self.stage_name != 'test':
            loss_meter = AverageValueMeter()
            metrics_meters = {metric.__name__: AverageValueMeter()
                              for metric in self.metrics}
        if _logs is not None:
            _logs.clear()
        np_y_pred = []

        with tqdm(dataloader, desc=self.stage_name,
                  file=sys.stdout, disable=not self.verbose) as iterator:
            for item in iterator:
                x = item[inp_index]
                y = item[gt_index]
                x, y = x.to(self.device), y.to(self.device)
                if _tta:
                    losses = []
                    preds = []
                    for cnt, (x_, y_) in enumerate(zip(x, y)):
                        _loss, _pred = self.batch_update(x_, y_, logs=_logs)
                        losses.append(_loss)
                        if _tta_inv_transforms is not None:
                            _pred = _tta_inv_transforms[cnt](_pred)
                        preds.append(_pred)
                    loss = torch.mean(torch.stack(losses), dim=0)
                    if _tta_strategy == 'mean':
                        y_pred = torch.mean(torch.stack(preds), dim=0)
                    elif _tta_strategy == 'max':
                        y_pred = torch.max(torch.stack(preds), dim=0)
                    else:
                        raise ValueError(f"Provided TTA strategy ({_tta_strategy})"
                                         " was not understood, `mean` or `max` currently handled")
                else:
                    loss, y_pred = self.batch_update(x, y, logs=_logs)

                if self.stage_name != 'test':
                    # update loss logs
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    loss_logs = {self.loss.__name__: loss_meter.mean}
                    logs.update(loss_logs)

                    # update metrics logs
                    for metric_fn in self.metrics:
                        metric_value = metric_fn(
                            y_pred, y).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k,
                                    v in metrics_meters.items()}
                    logs.update(metrics_logs)
                    try:
                        logs.update(
                            {'LR': self.optimizer.param_groups[0]['lr']})
                    except AttributeError:
                        pass
                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)
                else:
                    np_y_pred.append(y_pred.cpu().data.numpy())

        if self.stage_name == 'test':
            return np.concatenate(np_y_pred, axis=0)

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
        prediction = self.model(x)

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
            prediction = self.model(x)
            loss = self.loss(prediction, y, logs=logs)
        return loss, prediction


class TestEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='test',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, logs=None):
        with torch.no_grad():
            prediction = self.model(x)
        return None, prediction
