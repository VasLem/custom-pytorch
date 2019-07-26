import numpy as np
import os
import datetime
import torch
from torch.optim.lr_scheduler import _LRScheduler
from math import cos, pi
class AdaptiveCyclicCosineAnnealing(_LRScheduler):
    def __init__(self, initial_lr, optimizer, initial_cycle, batches_num,
                 tolerance=0.05,
                 local_minima_epochs_num=10, last_epoch=-1, annealing_snapshots_dir='', snapshots_num=4):
        self.annealing_snapshots_register_path = f'/tmp/{datetime.datetime.now()}annealing_models_reference.pkl'
        self.annealing_snapshots_num = snapshots_num
        self.batches_num = batches_num
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in self.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        assert len(self.optimizer.param_groups) == 1, len(
            self.optimizer.param_groups)
        self.base_lrs = list(
            map(lambda group: group['initial_lr'], self.optimizer.param_groups))
        self.last_epoch = last_epoch
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.adaptive_cycle = initial_cycle
        self.warm_restart_iteration = 1
        self.tolerance = tolerance
        self.current_iteration = 0
        self.history = np.zeros(local_minima_epochs_num)
        self.history[:] = np.nan
        self.annealing_snapshots_dir = annealing_snapshots_dir

    def identify_local_minima(self):
        """Compare last epochs_num loss and identify local minima existence
        """
        return np.abs(np.mean(np.diff(self.history))) <= self.tolerance

    def warm_restart(self, model, loss, error_metric):
        print('Warm restarting..')
        self.warm_restart_iteration = self.current_iteration
        self.history[:] = np.nan
        self.lr = self.initial_lr
        print('Saving annealing snapshot..')

        if self.current_iteration == 1:
            try:
                os.remove(self.annealing_snapshots_register_path)
            except BaseException:
                pass
        annealing_model_path = os.path.join(self.annealing_snapshots_dir,
            'SNAPSHOT_{}'.format(self.current_iteration))
        import pickle
        try:
            with open(self.annealing_snapshots_register_path, 'rb') as inp:
                annealing_snapshots_register = pickle.load(inp)
        except BaseException:
            annealing_snapshots_register = []
        to_save = False
        if len(annealing_snapshots_register) < self.annealing_snapshots_num:
            annealing_snapshots_register.append(
                (annealing_model_path, error_metric))
            to_save = True
        elif len(annealing_snapshots_register) == self.annealing_snapshots_num and any(
                error_metric < error for _, error in annealing_snapshots_register):
            errors = [error for _, error in annealing_snapshots_register]
            index = errors.index(max(errors))
            try:
                os.remove(annealing_snapshots_register[index][0])
            except BaseException:
                pass
            del annealing_snapshots_register[index]
            to_save = True
        if to_save:
            annealing_snapshots_register.append(
                (annealing_model_path, error_metric))
            with open(self.annealing_snapshots_register_path, 'wb') as inp:
                pickle.dump(annealing_snapshots_register, inp)
            torch.save({
                'iteration': self.current_iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss
            },  annealing_model_path)
        torch.cuda.empty_cache()
        for group in self.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

    def compute_lr(self, model, loss, error_metric):
        if not self.current_iteration % self.batches_num and self.identify_local_minima():
            self.adaptive_cycle = max(
                2, int((self.current_iteration - self.warm_restart_iteration) * 1.5) //
                self.batches_num * self.batches_num)
            print('New iterations cycle for learning rate warm restart:',
                  self.adaptive_cycle)
            self.warm_restart(model, loss, error_metric)

        else:
            self.lr = self.initial_lr / 2 * (cos(
                pi * ((self.current_iteration - self.warm_restart_iteration) %
                      self.adaptive_cycle) / self.adaptive_cycle) + 1)
            if self.current_iteration != self.warm_restart_iteration and\
                    abs(self.lr - self.initial_lr) <= 1e-15:
                self.warm_restart(model, loss, error_metric)

    def step(self, loss, model, error_metric, epoch=None):
        np_loss = np.mean(loss.cpu().data.numpy()) if loss is not None else 0
        if not np.isfinite(np_loss):
            print('Warning: Loss is None, reducing initial LR by 0.9 and warm restarting...')
            self.initial_lr = 0.9 * self.initial_lr
            self.warm_restart(model, loss, error_metric)
        else:
            self.current_iteration += 1
            if not self.current_iteration % self.batches_num:
                self.history[1:] = self.history[:-1]
                self.history[0] = np_loss
            if epoch is None:
                epoch = self.last_epoch + 1
            self.last_epoch = epoch
            self.compute_lr(model, loss, error_metric)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr