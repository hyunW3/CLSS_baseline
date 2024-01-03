import logging
import logging.config
from pathlib import Path
from utils.utils import read_json
import wandb

class Logger:
    def __init__(
        self,
        logdir,
        rank,
    ):
        self.rank = rank
        self.logger = None

        setup_logging(logdir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def set_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_levels[verbosity])
    def set_wandb(self, config):
        if self.rank == 0:
            project_name = "CLSS"
            if config['method'] == "base":
                project_name = "CLSS_new-method"
                self.wandb = wandb.init(
                    project=project_name,
                    tags=["finetune", config['method'],f"use_cosine{config['trainer']['use_cosine']}"],
                )
            self.wandb = wandb.init(
                project="CLSS",
                tags=[config['method']],
            )
            method = f"_{config['name']}" if config['method'] not in  config['name'] else ""
            wandb.run.name = config['data_loader']['args']['task']['setting'] + '_' \
            + config['data_loader']['args']['task']['name'] + '_'\
            + config['name'] + method + f"_step{config['data_loader']['args']['task']['step']}"

    # Print from all rank
    def print(self, msg):
        self.logger.info(msg)

    # Print from rank0 process only
    def info(self, msg):
        if self.rank == 0:
            self.logger.info(msg)

    def debug(self, msg):
        if self.rank == 0 and self.debug_flag:
            self.logger.info(msg)

    def error(self, msg):
        if self.rank == 0:
            self.logger.error(msg)

    def warning(self, msg):
        if self.rank == 0:
            self.logger.warning(msg)
    def log_wandb(self, data : dict, step = None):
        if self.rank == 0:
            self.wandb.log(data, step=step)
    def saveconfig_wandb(self, config):
        if self.rank == 0:
            self.wandb.config.update(config)
    def watch_wandb(self, model):
        if self.rank == 0:
            self.wandb.watch(model, log="all",log_graph=True)
def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])
                
        logging.config.dictConfig(config)

    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
