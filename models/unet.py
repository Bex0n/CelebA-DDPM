from typing import List

import numpy as np
import pytorch_lightning as pl
import torch


class DownsampleBlock(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.norm = torch.nn.LayerNorm(out_channels)
        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.norm(self.conv(x))))


class UpsampleBlock(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.norm = torch.nn.LayerNorm(out_channels)
        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.norm(self.conv(x))))


class UNet(pl.LightningModule):
    def __init__(
        self, channels: int = 128, depth: int = 3, dropout: float = 0.1, *args, **kwargs
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout

        self.downsample_blocks = torch.nn.ModuleList()
        self.upsample_blocks = torch.nn.ModuleList()

        for _ in range(depth):
            self.downsample_blocks.append(
                DownsampleBlock(channels, channels, dropout=dropout)
            )
            self.upsample_blocks.append(
                UpsampleBlock(channels, channels, dropout=dropout)
            )

        self.final_conv = torch.nn.Conv2d(channels, 3, 1)

    def forward(self, x):
        downsampled = []
        for block in self.downsample_blocks[:-1]:
            x = block(x)
            downsampled.append(x)
            x = torch.nn.functional.max_pool2d(x, 2)

        x = self.downsample_blocks[-1](x)

        for block in self.upsample_blocks:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
            x = torch.cat([x, downsampled.pop()], dim=1)
            x = block(x)

        return self.final_conv(x)

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
