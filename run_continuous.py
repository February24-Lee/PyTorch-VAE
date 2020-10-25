import yaml
import argparse
import numpy as np
from collections import OrderedDict

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

parser.add_argument('--checkpoint', '-ck',
                    dest="checkpoint",
                    metavar='FILE')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

#checkpoint load
#checkpoint = torch.load(args.checkpoint)
#new_state_dict = OrderedDict()
#for k,v in checkpoint['state_dict'].items():
#    name = k[6:]
#    new_state_dict[name] = v
#checkpoint['state_dict'] = new_state_dict

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

# MODEL
model = vae_models[config['model_params']['name']](**config['model_params'])

# 
experiment = VAEXperiment(model,
                          config['exp_params'])

runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                resume_from_checkpoint=args.checkpoint,
                 logger=tt_logger,
                 row_log_interval=1,
                 log_save_interval=100,
                 #train_percent_check=1.,
                 #val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)