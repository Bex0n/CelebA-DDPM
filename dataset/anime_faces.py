import pytorch_lightning as pl
from datasets import load_dataset


class AnimeFacesDataset(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = "dataset/anime-faces", batch_size: int = 32
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        self.dataset = load_dataset("huggan/anime-faces", data_dir=self.data_dir)
        print(self.dataset)

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
