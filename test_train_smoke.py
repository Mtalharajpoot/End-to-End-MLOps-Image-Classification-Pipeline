"""End-to-end smoke test: train 1 epoch on fake data (CPU), export ONNX, run inference."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")
pytest.importorskip("timm")
pytest.importorskip("onnx")

from PIL import Image  # noqa: E402

from visionops.config import load_config  # noqa: E402
from visionops.inference import Predictor  # noqa: E402
from visionops.train import run  # noqa: E402

FAST_DEV = Path(__file__).parents[1] / "configs" / "fast_dev.yaml"


def test_train_export_predict(tmp_path):
    cfg = load_config(FAST_DEV, {"output_dir": str(tmp_path), "data_dir": str(tmp_path)})
    results = run(cfg)

    assert 0.0 <= results["test_acc"] <= 1.0
    assert Path(results["onnx_path"]).exists()
    assert (tmp_path / "results.json").exists()

    preds = Predictor(results["onnx_path"]).predict(Image.new("RGB", (50, 50)), top_k=3)
    assert len(preds) == 3
