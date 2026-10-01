import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("timm")

from visionops.model import LitClassifier  # noqa: E402


def test_forward_shape():
    model = LitClassifier(backbone="resnet18", pretrained=False, num_classes=10)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_loss_is_finite_and_differentiable():
    model = LitClassifier(backbone="resnet18", pretrained=False, num_classes=10)
    batch = (torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))
    loss, _, _ = model._shared_step(batch)
    loss.backward()
    assert torch.isfinite(loss)
