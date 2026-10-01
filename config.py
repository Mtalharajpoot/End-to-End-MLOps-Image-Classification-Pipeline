"""Typed experiment configuration: YAML file + command-line overrides."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # --- data ---
    dataset: str = "cifar10"  # "cifar10" | "fake" (fake = offline smoke tests / CI)
    data_dir: str = "data"
    image_size: int = 64
    batch_size: int = 128
    num_workers: int = 2
    val_split: float = 0.1
    # --- model ---
    backbone: str = "resnet18"  # any timm model name
    pretrained: bool = True
    num_classes: int = 10
    # --- optimisation ---
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    # --- trainer / compute ---
    max_epochs: int = 5
    precision: str = "16-mixed"  # falls back to 32-true on CPU
    accelerator: str = "auto"
    devices: str = "auto"  # "auto" or a number such as "2"
    strategy: str = "auto"  # "auto" | "ddp" | "fsdp" | "deepspeed_stage_2" ...
    seed: int = 42
    patience: int = 3
    # --- tracking & outputs ---
    logger: str = "csv"  # "csv" | "wandb"
    project: str = "visionops"
    run_name: str = ""
    output_dir: str = "outputs"
    export_onnx: bool = True


def _coerce(raw: Any, default: Any) -> Any:
    """Cast a YAML / CLI value to the type of the field's default."""
    if isinstance(default, bool):
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return str(raw)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Build a Config from defaults < YAML file < overrides (``None`` overrides are ignored)."""
    values: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        values.update(yaml.safe_load(path.read_text()) or {})
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})

    defaults = Config()
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return Config(**{k: _coerce(v, getattr(defaults, k)) for k, v in values.items()})
