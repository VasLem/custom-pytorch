import os
from custom_pytorch.custom_utils import get_model_name
from custom_pytorch.custom_config import Config


class Logger:
    def __init__(self, config: Config, save_dir='logs', create_dir=False):
        """

        :param config: The configuration used
        :type config: Config
        :param save_dir: the directory to save the logs
        :type save_dir: path
        :param create_dir: whether to create supplied directory,
            will ignore this flag and automatically create if default directory is provided,
             defaults to False
        :type crate_dir: bool
        """
        if (create_dir or save_dir == 'logs') and not os.path.isdir(save_dir):
            print(f"Creating logs directory: {save_dir}")
            os.makedirs(save_dir)
        if not os.path.isdir(save_dir):
            raise ValueError(
                f"Provided directory \"{save_dir}\" is not an existing directory")
        self.save_dir = save_dir
        self.model_name = get_model_name(config)
        self.train_logs = os.path.join(
            self.save_dir, f'{self.model_name}_train.csv')
        self.valid_logs = os.path.join(
            self.save_dir, f'{self.model_name}_valid.csv')

    def update(self, step, logs, valid=False):
        if not valid:
            logs_fil = self.train_logs
        else:
            logs_fil = self.valid_logs
        if not os.path.isfile(logs_fil):
            with open(logs_fil, 'w') as myfile:
                myfile.write(
                    "Step\t" + '\t'.join([key for key in sorted(logs)]) + '\n')
        dat = f"{step}\t" + '\t'.join([str(logs[key])
                                       for key in sorted(logs)]) + '\n'

        with open(logs_fil, "a") as myfile:
            myfile.write(dat)
