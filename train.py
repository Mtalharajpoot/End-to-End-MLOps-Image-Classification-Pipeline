"""Training entry point: ``python -m visionops.train --config configs/default.yaml --lr 3e-4``."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from visionops.config import Config, load_config
from visionops.data import CLASSES, CIFAR10DataModule
from visionops.export_onnx import export_onnx
from visionops.model import LitClassifier


def resolve_precision(precision: str, cuda_available: bool) -> str:
    """Mixed precision needs a GPU; fall back to full precision on CPU."""
    if not cuda_available and precision.startswith(("16", "bf16")):
        warnings.warn(f"precision={precision!r} needs a GPU; using '32-true' on CPU.", stacklevel=2)
        return "32-true"
    return precision


def parse_devices(devices: str) -> int | str:
    return int(devices) if devices.isdigit() else devices


def build_logger(cfg: Config, out_dir: Path):
    if cfg.logger == "wandb":
        try:
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(
                project=cfg.project, name=cfg.run_name or None, save_dir=str(out_dir)
            )
            logger.log_hyperparams(asdict(cfg))
            return logger
        except ImportError:
            warnings.warn(
                "wandb not installed (pip install wandb); using CSV logger.", stacklevel=2
            )
    return CSVLogger(save_dir=str(out_dir), name=cfg.project, version=cfg.run_name or None)


def run(cfg: Config) -> dict[str, Any]:
    """Train, evaluate on the test set with the best checkpoint, and export to ONNX."""
    L.seed_everything(cfg.seed, workers=True)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = CIFAR10DataModule(
        dataset=cfg.dataset,
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )
    model = LitClassifier(
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        label_smoothing=cfg.label_smoothing,
        image_size=cfg.image_size,
    )

    checkpoint = ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1, filename="best-{epoch:02d}-{val_acc:.3f}"
    )
    callbacks = [
        checkpoint,
        EarlyStopping(monitor="val_loss", mode="min", patience=cfg.patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=parse_devices(cfg.devices),
        strategy=cfg.strategy,
        precision=resolve_precision(cfg.precision, torch.cuda.is_available()),
        logger=build_logger(cfg, out_dir),
        callbacks=callbacks,
        default_root_dir=str(out_dir),
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=dm)
    test_metrics = trainer.test(model, datamodule=dm, ckpt_path="best")[0]

    results: dict[str, Any] = {
        "test_acc": float(test_metrics["test_acc"]),
        "test_loss": float(test_metrics["test_loss"]),
        "best_checkpoint": checkpoint.best_model_path,
        "log_dir": getattr(trainer.logger, "log_dir", None),
        "onnx_path": None,
        "config": asdict(cfg),
    }
    if trainer.is_global_zero:
        if cfg.export_onnx and checkpoint.best_model_path:
            best = LitClassifier.load_from_checkpoint(
                checkpoint.best_model_path, pretrained=False, map_location="cpu"
            )
            onnx_path = export_onnx(best.model, out_dir / "model.onnx", cfg.image_size, CLASSES)
            results["onnx_path"] = str(onnx_path)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a VisionOps image classifier.")
    parser.add_argument("--config", default=None, help="YAML config (defaults are built in)")
    for f in fields(Config):  # every config field can be overridden: --lr 3e-4 or --lr=3e-4
        parser.add_argument(f"--{f.name}", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = vars(build_parser().parse_args(argv))
    cfg = load_config(args.pop("config"), args)
    results = run(cfg)
    print(f"\nTest accuracy: {results['test_acc']:.4f} | ONNX: {results['onnx_path']}")


if __name__ == "__main__":
    main()
