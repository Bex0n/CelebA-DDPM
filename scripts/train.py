import argparse

import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataset.anime_faces import AnimeFacesDataset
from models.diffusion import Diffusion


def train(conf: omegaconf.DictConfig, ckpt_dir: str):
    dataset = AnimeFacesDataset()
    dataloader = DataLoader(dataset, batch_size=conf.params.batch_size, shuffle=True)
    model = Diffusion(conf)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="diffusion-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        every_n_epochs=conf.params.save_every_n_epochs,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=conf.params.num_epochs,
        log_every_n_steps=conf.params.log_every_t,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, default="config.yaml")
    parser.add_argument("--ckpt_dir", type=str, required=True, default="dataset")
    args = parser.parse_args()

    conf = omegaconf.OmegaConf.load(args.config)
    train(conf, args.ckpt_dir)
