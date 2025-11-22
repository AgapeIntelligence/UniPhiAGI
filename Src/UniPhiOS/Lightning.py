"""
PyTorch Lightning wrapper for training/validation of GenesisGeometry.
This module provides a minimal LightningModule suitable for training tests.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from .engine import GenesisGeometry


class UniPhiLightning(pl.LightningModule):
    def __init__(self, device="cpu", dtype=torch.float32, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters({"device": device, "dtype": str(dtype), "lr": lr})
        self.model = GenesisGeometry(device=device, dtype=dtype)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        identity = batch["identity"].to(device=self.model.device, dtype=self.model.dtype)
        bloom, identity_next, _, _, spiral = self.model(identity)
        invariant = self.model.lattice_collapse(spiral)
        norm = identity_next.norm(dim=1).mean() if identity_next.dim() == 2 else identity_next.norm()
        loss = self.model.compute_toroidal_loss(bloom, invariant, norm, kl_div=0.0008)
        loss_scalar = loss.mean()
        self.log("train_loss", loss_scalar, prog_bar=True)
        return loss_scalar

    def validation_step(self, batch, batch_idx):
        identity = batch["identity"].to(device=self.model.device, dtype=self.model.dtype)
        bloom, identity_next, _, _, spiral = self.model(identity)
        invariant = self.model.lattice_collapse(spiral)
        norm = identity_next.norm(dim=1).mean() if identity_next.dim() == 2 else identity_next.norm()
        loss = self.model.compute_toroidal_loss(bloom, invariant, norm, kl_div=0.0008)
        self.log("val_loss", loss.mean())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
