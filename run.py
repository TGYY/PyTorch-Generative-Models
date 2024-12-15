import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment, LogSampleCallback
import torch.backends.cudnn as cudnn
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from dataset import VAEDataset
import wandb
from datetime import datetime

wandb.login()

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument(
    '--config', '-c',
    dest="filename",
    metavar='FILE',
    help='path to the config file',
    default='configs/vq_vae.yaml'
)

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Initialize the WandbLogger
wandb_logger = WandbLogger(
    save_dir=config['logging_params']['save_dir'],
    project=config['logging_params'].get('name', 'default_project'),
    log_model=True  
)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], workers=True)

pin_memory = True
data = VAEDataset(**config["data_params"], pin_memory=pin_memory)
data.setup()
config['model_params']['dataset_var'] = data.dataset_var

model = vae_models[config['model_params']['name']](**config['model_params'])
wandb_logger.log_text(key="ML", columns=["Model Structure"], data=[[str(model)]])

combined_config = {**config['model_params'], **config['exp_params'], **config['data_params']}
experiment = VAEXperiment(model, combined_config)


checkpoint_dir = os.path.join(
    config['logging_params']['save_dir'],
    f"{config['model_params']['name']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    "checkpoints"
)

os.makedirs(checkpoint_dir, exist_ok=True)

sample_save_dir = os.path.join(config['logging_params']['save_dir'], f"{config['model_params']['name']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
log_samples_callback = LogSampleCallback(log_dir=sample_save_dir)

runner = Trainer(
    logger=wandb_logger,
    callbacks=[
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch}-{val_loss:.2f}',
            save_top_k=2,
            monitor="val_loss",
            save_last=True,
        ),
        log_samples_callback,
    ],
    strategy=DDPStrategy(find_unused_parameters=False),
    **config['gpu_params'],
    max_epochs=10,
    # **config['test_run_config']
)

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
