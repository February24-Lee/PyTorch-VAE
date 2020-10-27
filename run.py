import yaml
import argparse
import numpy as np
import os
from pathlib import Path

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger, CometLogger


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

parser.add_argument('--comet_api', '-ca',
                    dest='comet_api')


args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


with open(args.comet_api, 'r') as f:
    try : 
        comet_api = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# --- TestTubeLogger
tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# --- CometLogger
comet_logger = CometLogger(
    api_key=comet_api['comet_ml_api'],
    **config['comet_params']
)


# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                 logger=comet_logger,
                 row_log_interval=1,
                 log_save_interval=100,
                 #train_percent_check=1.,
                 #val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)