from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch

from models.unet import UNet


class Diffusion(pl.LightningModule):
    def __init__(self, conf: Dict[str, Any]):
        super().__init__()
        self.conf = conf
        self.model = UNet(**conf)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.conf.params.lr,
            betas=self.conf.params.optimizer_config.params.betas,
            eps=self.conf.params.optimizer_config.params.eps,
            weight_decay=self.conf.params.optimizer_config.params.weight_decay,
        )
