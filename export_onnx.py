"""Export a trained checkpoint to ONNX and verify numerical parity with PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from visionops.data import CLASSES, MEAN, STD
from visionops.inference import meta_path
from visionops.model import LitClassifier


def export_onnx(
    net: torch.nn.Module,
    out_path: str | Path,
    image_size: int,
    classes: tuple[str, ...] = CLASSES,
    opset: int = 17,
    atol: float = 1e-3,
) -> Path:
    """Export ``net`` (plain nn.Module) to ONNX with a dynamic batch axis, then check parity."""
    import onnxruntime as ort

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    net = net.eval().cpu()
    dummy = torch.randn(1, 3, image_size, image_size)
    kwargs = {
        "input_names": ["image"],
        "output_names": ["logits"],
        "dynamic_axes": {"image": {0: "batch"}, "logits": {0: "batch"}},
        "opset_version": opset,
    }
    with torch.no_grad():
        try:  # torch >= 2.5: use the stable TorchScript-based exporter
            torch.onnx.export(net, dummy, str(out_path), dynamo=False, **kwargs)
        except TypeError:  # older torch has no `dynamo` argument
            torch.onnx.export(net, dummy, str(out_path), **kwargs)

    meta_path(out_path).write_text(
        json.dumps(
            {
                "classes": list(classes),
                "image_size": image_size,
                "mean": list(MEAN),
                "std": list(STD),
                "opset": opset,
            },
            indent=2,
        )
    )

    # Parity check: ONNX Runtime output must match PyTorch.
    batch = torch.randn(4, 3, image_size, image_size)
    with torch.no_grad():
        expected = net(batch).numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"image": batch.numpy()})[0]
    max_diff = float(np.abs(expected - got).max())
    if max_diff > atol:
        raise RuntimeError(f"ONNX parity check failed: max |diff| = {max_diff:.2e} > {atol:.0e}")
    print(f"ONNX exported to {out_path} (parity max |diff| = {max_diff:.2e})")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Export a Lightning checkpoint to ONNX.")
    p.add_argument("--ckpt", required=True, help="Path to a .ckpt file")
    p.add_argument("--out", default="models/model.onnx")
    args = p.parse_args()
    lit = LitClassifier.load_from_checkpoint(args.ckpt, pretrained=False, map_location="cpu")
    export_onnx(lit.model, args.out, image_size=lit.hparams.image_size)


if __name__ == "__main__":
    main()
