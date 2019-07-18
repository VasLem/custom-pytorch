import os
from custom_pytorch.custom_utils import get_model_name
from custom_pytorch.custom_config import Config


class Logger:
    def __init__(self, config: Config, metric_used, save_dir):
        """

        :param config: The configuration used
        :type config: Config
        :param metric_used: the metric used
        :type metric_used: str
        :param save_dir: the directory to save the logs
        :type save_dir: path
        """
        assert os.path.isdir(
            save_dir), "Provided directory is not a valid path"
        self.save_dir = save_dir
        self.metric = metric_used
        self.model_name = get_model_name(config)
        self.train_loss_logs = open(
            os.path.join(self.save_dir, 'train_loss_{self.model_name}.txt'), 'w')
        self.train_metric_logs = open(
            os.path.join(self.save_dir, 'train_{metric}_{self.model_name}.txt'), 'w')
        self.valid_loss_logs = open(
            os.path.join(self.save_dir, 'valid_loss_{self.model_name}.txt'), 'w')
        self.valid_metric_logs = open(
            os.path.join(self.save_dir, 'valid_{metric}_{self.model_name}.txt'), 'w')

    def update(self, step, loss, metric, valid=False):
        if not valid:
            loss_logs = self.train_loss_logs
            metric_logs = self.train_metric_logs
        else:
            loss_logs = self.valid_loss_logs
            metric_logs = self.valid_metric_logs
        loss_logs.write(f"{step},{loss}\n")
        metric_logs.write(f"{step},{metric}\n")
