"""Torch-free inference with ONNX Runtime (this is what the Docker serving image runs)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def meta_path(onnx_path: str | Path) -> Path:
    p = Path(onnx_path)
    return p.with_name(f"{p.stem}_meta.json")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class Predictor:
    """Loads ``model.onnx`` plus ``model_meta.json`` (classes, image size, normalisation)."""

    def __init__(self, onnx_path: str | Path) -> None:
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        meta = json.loads(meta_path(self.onnx_path).read_text())
        self.classes: list[str] = meta["classes"]
        self.image_size: int = meta["image_size"]
        self.mean = np.array(meta["mean"], dtype=np.float32)
        self.std = np.array(meta["std"], dtype=np.float32)
        self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])

    def preprocess(self, image: Image.Image) -> np.ndarray:
        size = self.image_size
        img = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return arr.transpose(2, 0, 1)[None].astype(np.float32)

    def predict(self, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        logits = self.session.run(None, {"image": self.preprocess(image)})[0][0]
        probs = softmax(logits)
        order = np.argsort(-probs)[:top_k]
        return [(self.classes[i], float(probs[i])) for i in order]
