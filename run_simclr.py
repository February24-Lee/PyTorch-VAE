import yaml
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from simCLR import *
from src import *

parser = argparse.ArgumentParser(description='Generic runner for simclr models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Callback function
checkpoint_callback_valid = ModelCheckpoint(monitor='avg_val_loss',
                                            save_last=True,
                                            save_top_k=1,
                                            mode='min')

# Logger
tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

datamodule = datamodule_model[config['exp_parmas']['dataset']](**config['exp_parmas'])

model = simclr_model[config['model_params']['name']](num_samples=datamodule.train_num ,**config['model_params'])

runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                 logger=tt_logger,
                 row_log_interval=10,
                 log_save_interval=100,
                 #train_percent_check=1.,
                 #val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 checkpoint_callback=checkpoint_callback_valid,
                 **config['trainer_params'],)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(model, datamodule)