import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice

from custom_pytorch.custom_config import Config
from custom_pytorch.custom_utils import get_model_name


class Visualizer:
    def __init__(self, config: Config, max_images_num=10,
                 metric_used='DiceCoeff', include_lr=True, examples_savedir=None):
        """Visualizer of segmentation specific problems

        :param config: the experiment configuration
        :type config: Config
        :param max_images_num: the maximum images to show, defaults to 10
        :type max_images_num: int, optional
        :param metric_used: the metric used by the experiment, defaults to 'DiceCoeff'
        :type metric_used: str, optional
        :param include_lr: whether to include learning rate in the plots, defaults to True
        :type include_lr: bool, optional
        :param examples_savedir: if provided, it is the path where the
            examples plots will be saved, defaults to None
        :type examples_savedir: path, optional
        """
        self.model_name = get_model_name(config)
        assert examples_savedir is None or os.path.isdir(examples_savedir),\
            'Examples savedir is not a valid directory'
        self.plot_num = min(config.batch_size, max_images_num)
        self.metric = metric_used
        self.examples_savedir = examples_savedir
        self.include_lr = include_lr
        self._lines_plot = None
        self._examples_plot = None
        self._lines = None

    def get_interactive_plot(self, titles):
        fig, axes = plt.subplots(nrows=len(titles), sharex=True)
        for title, axe in zip(titles, axes):
            axe.title.set_text(title)
            axe.grid(True, 'both')
        return fig, axes

    @property
    def lines_plot(self):
        if self._lines_plot is None:
            titles = ['Loss', self.metric, 'LR']
            if not self.include_lr:
                titles = titles[:-1]
            fig, axes = self.get_interactive_plot(titles)
            self._lines_plot = fig, axes
        return self._lines_plot

    @property
    def examples_plot(self):
        if self._examples_plot is None:
            fig, axes = plt.subplots(
                ncols=3, sharey=True, nrows=self.plot_num, sharex=True)
            axes[0][0].title.set_text('Input')
            axes[0][1].title.set_text('GroundTruth')
            axes[0][2].title.set_text('Predicted')
            self._examples_plot = fig, axes
        return self._examples_plot

    def update_line(self, hl, x, y):
        hl.set_xdata(np.append(hl.get_xdata(), x))
        hl.set_ydata(np.append(hl.get_ydata(), y))

    def check_and_update_line(self, ax, line, func, x, y, *args, **kwargs):

        if line is None:
            line = func(x, y, *args, **kwargs)
            ax.legend()
        else:
            self.update_line(line, x, y)

        ax.relim()
        ax.autoscale_view()
        return line

    @property
    def lines(self):
        if self._lines is None:
            self._lines = {'loss': {'train': None, 'valid': None},
                           'metric': {'train': None, 'valid': None}, 'lr': None}
        return self._lines

    def update_examples_plot(self, step, input_ims, input_masks, predicteds):
        fig, axes = self.examples_plot
        if len(input_ims) < self.plot_num:
            inds = np.arange(len(input_ims))
        else:
            inds = choice(range(len(input_ims)), self.plot_num, replace=False)
        for cnt, (input_im, input_mask, predicted) in enumerate(zip(input_ims[inds],
                                                                    input_masks[inds],
                                                                    predicteds[inds])):
            axes[cnt][0].imshow(input_im.squeeze())
            axes[cnt][1].imshow(input_mask.squeeze())
            axes[cnt][2].imshow(predicted.squeeze())
        fig.canvas.draw()
        fig.canvas.flush_events()
        if self.examples_savedir:
            fig.savefig(os.path.join(self.examples_savedir,
                                     f'{self.model_name}_{step}.jpg'))

    def update_lines_plot(self, train_x=None, train_loss=None, train_metric=None,
            valid_x=None, valid_loss=None, valid_metric=None, lr=None):
        fig, axes = self.lines_plot
        if train_x is not None:
            if train_loss is not None:
                self.lines['loss']['train'] = self.check_and_update_line(
                    axes[0],
                    self.lines['loss']['train'],
                    lambda *args, **kwargs: axes[0].plot(*args, **kwargs)[0],
                    train_x, train_loss, 'r-', label='Train')
            if train_metric is not None:
                self.lines['metric']['train'] = self.check_and_update_line(
                    axes[1],
                    self.lines['metric']['train'],
                    lambda *args, **kwargs: axes[1].plot(*args, **kwargs)[0],
                    train_x, train_metric, 'r-', label='Train')

        if valid_x is not None:
            if valid_loss is not None:
                self.lines['loss']['valid'] = self.check_and_update_line(
                    axes[0],
                    self.lines['loss']['valid'],
                    lambda *args, **kwargs: axes[0].plot(*args, **kwargs)[0],
                    valid_x, valid_loss, 'b-', label='Valid')
            if valid_metric is not None:
                self.lines['metric']['valid'] = self.check_and_update_line(
                    axes[1],
                    self.lines['metric']['valid'],
                    lambda *args, **kwargs: axes[1].plot(*args, **kwargs)[0],
                    valid_x, valid_metric, 'b-', label='Valid')
        if lr is not None:
            assert self.include_lr, 'Learning rate cannot be plotted with include_lr being false'
            self.lines['lr'] = self.check_and_update_line(
                axes[2], self.lines['lr'],
                lambda *args, **kwargs: axes[2].plot(*args, **kwargs)[0],
                train_x, lr, 'b-', label='LR')

        fig.canvas.draw()
        fig.canvas.flush_events()

    def update_loss(self, step, loss, valid=False):
        if valid:
            self.update_lines_plot(valid_x=step, valid_loss=loss)
        else:
            self.update_lines_plot(train_x=step, train_loss=loss)

    def update_metric(self, step, metric, valid=False):
        if valid:
            self.update_lines_plot(valid_x=step, valid_metric=metric)
        else:
            self.update_lines_plot(train_x=step, train_metric=metric)

    def update_loss_and_metric(self, step, loss, metric, valid=False):
        self.update_loss(step, loss, valid)
        self.update_metric(step, metric, valid)

    def update_lr(self, step, lr, valid=False):
        self.update_lines_plot(train_x=step, lr=lr)
