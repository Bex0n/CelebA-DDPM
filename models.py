import pytorch_lightning as pl
import torch

import torch.nn as nn

class DDPM(pl.LightningModule):
    def __init__(self, model, data_module, config):
        super().__init__()
        self.model = model
        self.data_module = data_module
        self.config = config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.config.adam_b1, self.config.adam_b2)
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch['image']
        loss = self.model(x)
        return loss

class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
