"""Data module: CIFAR-10 (or an offline fake dataset) with train/val/test splits."""

from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    ops: list = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [
            transforms.RandomCrop(image_size, padding=max(image_size // 8, 1)),
            transforms.RandomHorizontalFlip(),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: str = "cifar10",
        data_dir: str = "data",
        image_size: int = 64,
        batch_size: int = 128,
        num_workers: int = 2,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None
        self.test_ds: Dataset | None = None

    def _make(self, train: bool, augment: bool, offset: int = 0) -> Dataset:
        hp = self.hparams
        tf = build_transforms(hp.image_size, augment)
        if hp.dataset == "fake":
            size = 256 if train else 64
            return datasets.FakeData(
                size=size,
                image_size=(3, 32, 32),
                num_classes=10,
                transform=tf,
                random_offset=offset,
            )
        if hp.dataset == "cifar10":
            return datasets.CIFAR10(hp.data_dir, train=train, transform=tf)
        raise ValueError(f"Unknown dataset: {hp.dataset!r}")

    def prepare_data(self) -> None:
        if self.hparams.dataset == "cifar10":
            datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
            datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        hp = self.hparams
        train_full = self._make(train=True, augment=True)
        val_full = self._make(train=True, augment=False)  # same images, no augmentation
        n = len(train_full)
        n_val = int(n * hp.val_split)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(hp.seed)).tolist()
        self.train_ds = Subset(train_full, perm[n_val:])
        self.val_ds = Subset(val_full, perm[:n_val])
        self.test_ds = self._make(train=False, augment=False, offset=1000)

    def _loader(self, ds: Dataset | None, shuffle: bool) -> DataLoader:
        assert ds is not None, "Call setup() first"
        workers = self.hparams.num_workers
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
