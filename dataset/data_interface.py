import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import WSIDataset


class WSIDataModule(pl.LightningDataModule):

    def __init__(self, dataset_path, fold, dataset):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """
        super().__init__()

        self.data_root = dataset_path
        self.fold = fold
        self.dataset_name = dataset

    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """
        - count number of classes

        - build vocabulary

        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = WSIDataset(self.data_root, state='train', fold=self.fold, dataset=self.dataset_name)
            self.val_dataset = WSIDataset(self.data_root, state='test', fold=self.fold, dataset=self.dataset_name)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = WSIDataset(self.data_root, state='test', fold=self.fold, dataset=self.dataset_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=0, shuffle=False)
