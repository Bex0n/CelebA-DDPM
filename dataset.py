import pytorch_lightning as L

from datasets import load_dataset
from torch.utils.data import DataLoader


class AnimeFacesDataset(L.LightningDataModule):
    def __init__(self, batch_size, num_workers, data_dir):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir

    def prepare_data(self):
        self.train_dataset = load_dataset('anime_faces', split='train')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.train_dataset.map(self.preprocess, batched=True)
            self.train_dataset.set_format(type='torch', columns=['image'])
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)