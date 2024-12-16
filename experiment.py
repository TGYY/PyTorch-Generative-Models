import os
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import lightning.pytorch as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
import torch
from PIL import Image
import numpy as np

def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array[imgs_array < -0.5] = -0.5
    imgs_array[imgs_array > 0.5] = 0.5
    imgs_array = 255 * (imgs_array + 0.5)
    (batch_size, img_size) = img_tensors.shape[:2]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * batch_size, ncols * batch_size, 3))
    for idx in range(36):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict,
                 ) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.save_hyperparameters(ignore=['vae_model'])
        # self.hold_graph = False
        # try:
        #     self.hold_graph = self.params['retain_first_backpass']
        # except:
        #     pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['kld_weight'], # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_validation_end(self) -> None:
        pass
        # self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=6)

        try:
            samples = self.model.sample(36,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=6)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=0)
        optims.append(optimizer)
        # # Check if more than 1 optimizer is required (Used for adversarial training)
        # try:
        #     if self.params['LR_2'] is not None:
        #         optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
        #                                 lr=self.params['LR_2'])
        #         optims.append(optimizer2)
        # except:
        #     pass

        # try:
        #     if self.params['scheduler_gamma'] is not None:
        #         scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
        #                                                      gamma = self.params['scheduler_gamma'])
        #         scheds.append(scheduler)

        #         # Check if another scheduler is required for the second optimizer
        #         try:
        #             if self.params['scheduler_gamma_2'] is not None:
        #                 scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
        #                                                               gamma = self.params['scheduler_gamma_2'])
        #                 scheds.append(scheduler2)
        #         except:
        #             pass
        #         return optims, scheds
        # except:
        #     pass
        return optims


class LogSampleCallback(Callback):
    def __init__(self, log_dir, num_samples=36, nrow=6):
        super().__init__()
        self.log_dir = log_dir
        self.num_samples = num_samples
        self.nrow = nrow

    def on_validation_epoch_end(self, trainer, pl_module):
        # Ensure the necessary directories exist
        recon_dir = os.path.join(self.log_dir, "Reconstructions")
        sample_dir = os.path.join(self.log_dir, "Samples")
        os.makedirs(recon_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        # Move the model to the appropriate device
        curr_device = pl_module.device

        # Get a batch of test data
        test_loader = trainer.datamodule.test_dataloader()
        test_input, test_label = next(iter(test_loader))
        test_input = test_input.to(curr_device)
        test_label = test_label.to(curr_device)

        # Save test_input
        test_input_file = os.path.join(
            recon_dir,
            f"test_input_{trainer.logger.name}_Epoch_{trainer.current_epoch}.png"
        )
        vutils.save_image(
            test_input.data,
            test_input_file,
            normalize=True,
            nrow=self.nrow
        )

        test_input_imgs = list(torch.unbind(test_input, dim=0))

        # Log test_input to Wandb
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image(
                key="Test Input",
                images=test_input_imgs
            )

        # Generate reconstructions
        recons = pl_module.model.generate(test_input, labels=test_label)
        recon_file = os.path.join(
            recon_dir,
            f"recons_{trainer.logger.name}_Epoch_{trainer.current_epoch}.png"
        )
        vutils.save_image(
            recons.data,  # Normalize from [-1, 1] to [0, 255]
            recon_file,
            normalize=True,
            nrow=self.nrow
        )
        recons_imgs = list(torch.unbind(recons, dim=0))
        # Log reconstructions to Wandb
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image(
                key="Reconstructions",
                images=recons_imgs
            )

        save_img_tensors_as_grid(test_input.data, 6, "true")
        save_img_tensors_as_grid(recons.data, 6, "recon")
        # test_labels = [torch.nonzero(label).squeeze().tolist() for label in list(torch.unbind(test_label, dim=0))]
        # test_labels = [list(map(str, label)) for label in test_labels]
        # Generate samples
        # try:
        #     samples = pl_module.model.sample(self.num_samples, curr_device, labels=test_label)
        #     # sample_file = os.path.join(
        #     #     sample_dir,
        #     #     f"{trainer.logger.name}_Epoch_{trainer.current_epoch}.png"
        #     # )
        #     # vutils.save_image(
        #     #     samples.cpu().data,
        #     #     sample_file,
        #     #     normalize=True,
        #     #     nrow=self.nrow
        #     # )

        #     sample_imgs = list(torch.unbind(samples, dim=0))
        #     # Log samples to Wandb
        #     if isinstance(trainer.logger, WandbLogger):
        #         trainer.logger.log_image(
        #             key="Samples",
        #             images=sample_imgs,
        #             caption=test_labels
        #         )

        # except Warning:
        #     pass