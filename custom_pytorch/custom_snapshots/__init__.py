import os
from custom_pytorch.custom_config import Config
from custom_pytorch.custom_utils import get_model_name
import torch


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
    def __init__(self, trainer, save_dir='models', create_dir=False):
        self.trainer = trainer
        if (create_dir or save_dir == 'logs') and not os.path.isdir(save_dir):
            print(f"Creating snapshots directory: {save_dir}")
            os.makedirs(save_dir)
        if not os.path.isdir(save_dir):
            raise ValueError(
                f"Provided directory \"{save_dir}\" is not an existing directory")
        self.models_dir = save_dir

    def save(self, epoch, train_loss, valid_loss, train_metric, valid_metric,
             id: str):
        model_name = get_model_name(
            self.trainer.config, epoch, train_loss, valid_loss, train_metric, valid_metric)
        model_name += id
        torch.save(Snapshot(config=self.trainer.config, epoch=epoch,
                            model=self.trainer.model.state_dict(),
                            optimizer=self.trainer.optimizer.state_dict(),
                            losses={'train': train_loss, 'valid': valid_loss},
                            metrics={'train': train_metric, 'valid': valid_metric}),
                   os.path.join(self.models_dir, model_name) + '.pth')

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
