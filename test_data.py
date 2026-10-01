import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("torchvision")

from PIL import Image  # noqa: E402

from visionops.data import CIFAR10DataModule, build_transforms  # noqa: E402


def test_transform_output():
    x = build_transforms(48, train=True)(Image.new("RGB", (32, 32)))
    assert x.shape == (3, 48, 48)
    assert x.dtype == torch.float32


def test_fake_datamodule_splits_and_batches():
    dm = CIFAR10DataModule(dataset="fake", image_size=32, batch_size=16, num_workers=0)
    dm.setup()
    assert len(dm.train_ds) + len(dm.val_ds) == 256
    assert not set(dm.train_ds.indices) & set(dm.val_ds.indices)  # no train/val leakage
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (16, 3, 32, 32)
    assert y.shape == (16,)
