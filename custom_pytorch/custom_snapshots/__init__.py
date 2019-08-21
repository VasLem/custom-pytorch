import os
from custom_pytorch.custom_config import Config
from custom_pytorch.custom_utils import get_model_name
import torch
from custom_pytorch.custom_utils.train import Trainer


class Snapshot:
    """The main snapshot constructor
    """

    def __init__(self, *args, config: Config = None,
                 epoch: int = None,
                 model=None, optimizer=None, losses=None, metrics=None):
        assert config is not None
        assert epoch is not None
        assert model is not None
        assert optimizer is not None
        self.config = config
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.losses = losses
        self.metrics = metrics


class SnapshotsHandler:
    def __init__(self, trainer: Trainer,
                 save_dir='models', create_dir=False, max_snapshots_to_keep=5):
        """The models snapshots handler

        :param trainer: the trainer
        :type trainer: Trainer
        :param save_dir: the saving directory, defaults to 'models'
        :type save_dir: str, optional
        :param create_dir: whether to create directory if it does not exist and
            is not the default one, defaults to False
        :type create_dir: bool, optional
        :param max_snapshots_to_keep: the maximum snapshots to keep per id, defaults to 5
        :type max_snapshots_to_keep: int, optional
        :raises ValueError: If create_dir is False and sthe provided directory does not exist
        """
        self.trainer = trainer
        if (create_dir or save_dir == 'logs') and not os.path.isdir(save_dir):
            print(f"Creating snapshots directory: {save_dir}")
            os.makedirs(save_dir)
        if not os.path.isdir(save_dir):
            raise ValueError(
                f"Provided directory \"{save_dir}\" is not an existing directory")
        self.models_dir = save_dir
        self.max_snapshots_to_keep = max_snapshots_to_keep
        self.names = {}

    def save(self, epoch, train_loss, valid_loss, train_metric, valid_metric,
             id: str):
        model_name = get_model_name(
            self.trainer.config, epoch, train_loss, valid_loss, train_metric, valid_metric)
        model_name += id + '.pth'
        if id not in self.names:
            self.names[id] = []
        self.names[id].append(model_name)
        torch.save(Snapshot(config=self.trainer.config, epoch=epoch,
                            model=self.trainer.model.state_dict(),
                            optimizer=self.trainer.optimizer.state_dict(),
                            losses={'train': train_loss, 'valid': valid_loss},
                            metrics={'train': train_metric, 'valid': valid_metric}),
                   os.path.join(self.models_dir, model_name))
        if len(self.names[id]) > self.max_snapshots_to_keep:
            os.remove(os.path.join(
                self.models_dir, self.names[id][0]))
            self.names[id] = self.names[id][1:]

    def load(self, model_name):
        if not model_name.endswith('.pth'):
            model_name += '.pth'
        snapshot = torch.load(os.path.join(self.models_dir, model_name))
        if self.trainer.config != snapshot.config:
            import pandas as pd
            attributes = sorted(list(set(
                [attr for attr in dir(self.trainer.config) if not attr.startswith('__')
                 and not callable(getattr(self.trainer.config, attr))] +
                [attr for attr in dir(snapshot.config) if not attr.startswith('__')
                 and not callable(getattr(snapshot.config, attr))])))
            bad_attrs = {}
            for attr in attributes:
                try:
                    getattr(self.trainer.config, attr)
                except BaseException:
                    bad_attrs[attr] = {'current': '-',
                                       'loaded': getattr(snapshot.config, attr)}
                    continue
                try:
                    getattr(snapshot.config, attr)
                except BaseException:
                    bad_attrs[attr] = {
                        'loaded': '-', 'current': getattr(self.trainer.config, attr)}
                    continue
                if getattr(self.trainer.config, attr) != getattr(snapshot.config, attr):
                    bad_attrs[attr] = {'loaded': getattr(snapshot.config, attr),
                                       'current': getattr(self.trainer.config, attr)}
            print(
                "The following attributes of the configuration differ from the ones loaded:")
            from IPython.display import display
            display(pd.DataFrame.from_dict(bad_attrs, orient='index'))

        self.trainer.epoch = snapshot.epoch + 1
        self.trainer.optimizer.load_state_dict(snapshot.optimizer)
        self.trainer.model.load_state_dict(snapshot.model)
