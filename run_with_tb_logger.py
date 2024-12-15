import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.strategies import DDPStrategy

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument(
    '--config', '-c',
    dest="filename",
    metavar='FILE',
    help='path to the config file',
    default='configs/vae.yaml'
)

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['model_params']['name'],
)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], workers=True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

# Adjusted pin_memory setting
gpus = config['trainer_params'].get('gpus', 0)
if isinstance(gpus, (list, tuple)):
    pin_memory = len(gpus) > 0
elif isinstance(gpus, int):
    pin_memory = gpus > 0
else:
    pin_memory = False

data = VAEDataset(**config["data_params"], pin_memory=pin_memory)
data.setup()

runner = Trainer(
    logger=tb_logger,
    callbacks=[
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            monitor="val_loss",
            save_last=True,
        ),
    ],
    strategy=DDPStrategy(find_unused_parameters=False),
    **config['trainer_params'],
    fast_dev_run=False
)

Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)