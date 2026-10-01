"""LightningModule wrapping any timm backbone."""

from __future__ import annotations

import lightning as L
import timm
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy


class LitClassifier(L.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.hparams.label_smoothing)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self._shared_step(batch)
        self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self._shared_step(batch)
        self.val_acc(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, y = self._shared_step(batch)
        self.test_acc(logits, y)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.hparams.lr,
            total_steps=int(self.trainer.estimated_stepping_batches),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
