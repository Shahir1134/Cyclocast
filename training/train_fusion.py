"""
training/train_fusion.py
==========================
Full training loop for the multimodal fusion pipeline.

Frozen components (not trained):
  - SatelliteWrapper (ResNet18 cyclone classifier)
  - GraphCastWrapper (JAX runner is never called during training;
    pre-computed atmospheric features are loaded from disk)

Trainable components:
  - AtmosphericEncoder
  - MultimodalFusionModel
  - TrackHead
  - IntensityHead
  - ConfidenceHead

Usage
-----
    python training/train_fusion.py --config config/config.yaml

    # Resume from checkpoint
    python training/train_fusion.py --config config/config.yaml \
        --resume checkpoints/best_fusion.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when run as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.config_loader import load_config, get_device
from data.dataset import build_dataloaders
from models.satellite_wrapper import SatelliteWrapper
from models.atmospheric_encoder import AtmosphericEncoder
from models.fusion_model import MultimodalFusionModel
from models.track_head import TrackHead
from models.intensity_head import IntensityHead
from models.confidence_head import ConfidenceHead
from training.losses import FusionLoss
from training.metrics import MetricsAccumulator
from utils.feature_extraction import AtmosphericFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_fusion")


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

class FusionSystem(nn.Module):
    """
    Container that wires all trainable components together.

    The satellite wrapper is held as a plain attribute (not an nn.Module child)
    so its parameters are never registered for gradient computation.
    """

    def __init__(
        self,
        atm_encoder: AtmosphericEncoder,
        fusion_model: MultimodalFusionModel,
        track_head: TrackHead,
        intensity_head: IntensityHead,
        confidence_head: ConfidenceHead,
    ) -> None:
        super().__init__()
        self.atm_encoder = atm_encoder
        self.fusion_model = fusion_model
        self.track_head = track_head
        self.intensity_head = intensity_head
        self.confidence_head = confidence_head

    def forward(
        self,
        satellite_embedding: torch.Tensor,
        cyclone_probability: torch.Tensor,
        atm_6h: torch.Tensor,
        atm_12h: torch.Tensor,
        atm_18h: torch.Tensor,
        atm_24h: torch.Tensor,
        location_encoding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Encode each horizon with the shared atmospheric encoder
        atm_embeds = self.atm_encoder.encode_horizons(
            [atm_6h, atm_12h, atm_18h, atm_24h]
        )

        # Fuse
        shared = self.fusion_model(
            satellite_embedding=satellite_embedding,
            cyclone_probability=cyclone_probability,
            atm_embeddings=atm_embeds,
            location_encoding=location_encoding,
        )

        # Heads
        return {
            "track":      self.track_head(shared),
            "intensity":  self.intensity_head(shared),
            "confidence": self.confidence_head(shared),
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_system(cfg: dict, device: torch.device) -> Tuple[SatelliteWrapper, FusionSystem]:
    """Instantiate all models from config."""

    # Satellite wrapper (frozen)
    sat_wrapper = SatelliteWrapper(
        model_path=cfg["satellite_model_path"],
        num_classes=cfg["satellite"]["num_classes"],
        device=device,
    )

    # Atmospheric encoder
    crop_cfg = cfg["crop"]
    feat_cfg = cfg["atmospheric_encoder"]
    window_deg = crop_cfg["window_deg"]
    grid_res   = crop_cfg["grid_resolution"]
    n_spatial  = int(round(2 * window_deg / grid_res)) + 1  # e.g. 81

    extractor = AtmosphericFeatureExtractor(spatial_size=n_spatial)
    feat_dim = extractor.feature_dim(n_spatial)

    atm_encoder = AtmosphericEncoder(
        input_dim=feat_dim,
        hidden_dim=feat_cfg["hidden_dim"],
        output_dim=feat_cfg["output_dim"],
        dropout=feat_cfg["dropout"],
    )

    # Fusion model
    f_cfg = cfg["fusion"]
    fusion_model = MultimodalFusionModel(
        satellite_embed_dim=cfg["satellite"]["embedding_dim"],
        atm_embed_dim=feat_cfg["output_dim"],
        sat_proj_dim=f_cfg["satellite_proj_dim"],
        atm_proj_dim=f_cfg["atm_proj_dim"],
        shared_hidden_dim=f_cfg["shared_hidden_dim"],
        num_horizons=len(cfg["graphcast"]["forecast_horizons_h"]),
        use_location_embedding=f_cfg["use_location_embedding"],
        location_embed_dim=f_cfg["location_embed_dim"],
        dropout=f_cfg["dropout"],
    )

    # Heads
    h_cfg = cfg["heads"]
    track_head      = TrackHead(fusion_model.output_dim, hidden_dim=h_cfg["track"]["hidden_dim"])
    intensity_head  = IntensityHead(fusion_model.output_dim, hidden_dim=h_cfg["intensity"]["hidden_dim"])
    confidence_head = ConfidenceHead(fusion_model.output_dim, hidden_dim=h_cfg["confidence"]["hidden_dim"])

    system = FusionSystem(
        atm_encoder=atm_encoder,
        fusion_model=fusion_model,
        track_head=track_head,
        intensity_head=intensity_head,
        confidence_head=confidence_head,
    ).to(device)

    trainable = sum(p.numel() for p in system.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{trainable:,}")

    return sat_wrapper, system


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------

def train_epoch(
    sat_wrapper: SatelliteWrapper,
    system: FusionSystem,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: FusionLoss,
    device: torch.device,
    grad_clip: float,
    accumulator: MetricsAccumulator,
) -> Dict[str, float]:
    system.train()
    accumulator.reset()

    for batch in loader:
        # Move to device
        sat_img       = batch["satellite_image"].to(device)
        loc_enc       = batch["location_encoding"].to(device)
        atm_6h        = batch["atm_6h"].to(device)
        atm_12h       = batch["atm_12h"].to(device)
        atm_18h       = batch["atm_18h"].to(device)
        atm_24h       = batch["atm_24h"].to(device)
        target_track  = batch["target_track"].to(device)
        target_int    = batch["target_intensity"].to(device)
        current_lat   = batch["current_lat"].to(device)
        current_lon   = batch["current_lon"].to(device)

        # Extract satellite embedding (frozen — no_grad)
        with torch.no_grad():
            sat_out = sat_wrapper(sat_img)
        sat_emb  = sat_out["satellite_embedding"].to(device)
        sat_prob = sat_out["cyclone_probability"].to(device)

        # Forward
        optimizer.zero_grad()
        outputs = system(sat_emb, sat_prob, atm_6h, atm_12h, atm_18h, atm_24h, loc_enc)

        # Loss
        losses = loss_fn(
            pred_track=outputs["track"],
            pred_intensity=outputs["intensity"],
            pred_confidence=outputs["confidence"],
            target_track=target_track,
            target_intensity=target_int,
            current_lat=current_lat,
            current_lon=current_lon,
        )

        losses["loss_total"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(system.parameters(), grad_clip)
        optimizer.step()

        accumulator.update(
            pred_track=outputs["track"],
            pred_intensity=outputs["intensity"],
            pred_confidence=outputs["confidence"],
            true_track=target_track,
            true_intensity=target_int,
            current_lat=current_lat,
            current_lon=current_lon,
            batch_loss=losses["loss_total"].item(),
        )

    return accumulator.compute()


@torch.no_grad()
def eval_epoch(
    sat_wrapper: SatelliteWrapper,
    system: FusionSystem,
    loader: DataLoader,
    loss_fn: FusionLoss,
    device: torch.device,
    accumulator: MetricsAccumulator,
) -> Dict[str, float]:
    system.eval()
    accumulator.reset()

    for batch in loader:
        sat_img      = batch["satellite_image"].to(device)
        loc_enc      = batch["location_encoding"].to(device)
        atm_6h       = batch["atm_6h"].to(device)
        atm_12h      = batch["atm_12h"].to(device)
        atm_18h      = batch["atm_18h"].to(device)
        atm_24h      = batch["atm_24h"].to(device)
        target_track = batch["target_track"].to(device)
        target_int   = batch["target_intensity"].to(device)
        current_lat  = batch["current_lat"].to(device)
        current_lon  = batch["current_lon"].to(device)

        sat_out = sat_wrapper(sat_img)
        sat_emb  = sat_out["satellite_embedding"].to(device)
        sat_prob = sat_out["cyclone_probability"].to(device)

        outputs = system(sat_emb, sat_prob, atm_6h, atm_12h, atm_18h, atm_24h, loc_enc)

        losses = loss_fn(
            pred_track=outputs["track"],
            pred_intensity=outputs["intensity"],
            pred_confidence=outputs["confidence"],
            target_track=target_track,
            target_intensity=target_int,
            current_lat=current_lat,
            current_lon=current_lon,
        )

        accumulator.update(
            pred_track=outputs["track"],
            pred_intensity=outputs["intensity"],
            pred_confidence=outputs["confidence"],
            true_track=target_track,
            true_intensity=target_int,
            current_lat=current_lat,
            current_lon=current_lon,
            batch_loss=losses["loss_total"].item(),
        )

    return accumulator.compute()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    system: FusionSystem,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "metrics":    metrics,
        "model_state_dict":     system.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    logger.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: Path,
    system: FusionSystem,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> int:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    system.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    logger.info("Loaded checkpoint from '%s' (epoch %d).", path, epoch)
    return epoch


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config_path: str, resume_path: Optional[str] = None) -> None:
    cfg = load_config(config_path)
    device = get_device(cfg.get("device", "auto"))

    logger.info("Device: %s", device)
    torch.manual_seed(cfg.get("seed", 42))

    # ── Data ─────────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    root_dir = Path(data_cfg["root_dir"])

    crop_cfg = cfg["crop"]
    window_deg = crop_cfg["window_deg"]
    grid_res   = crop_cfg["grid_resolution"]
    n_spatial  = int(round(2 * window_deg / grid_res)) + 1

    extractor = AtmosphericFeatureExtractor(spatial_size=n_spatial)

    train_loader, val_loader, _ = build_dataloaders(
        labels_csv=root_dir / data_cfg["labels_file"],
        root_dir=root_dir,
        feature_extractor=extractor,
        batch_size=cfg["training"]["batch_size"],
        num_workers=data_cfg["num_workers"],
        train_frac=data_cfg["train_split"],
        val_frac=data_cfg["val_split"],
        seed=cfg.get("seed", 42),
    )

    # ── Models ────────────────────────────────────────────────────────────────
    sat_wrapper, system = build_system(cfg, device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    t_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        system.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_cfg["num_epochs"], eta_min=1e-6
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    lw = t_cfg["loss_weights"]
    loss_fn = FusionLoss(
        track_weight=lw["track"],
        intensity_weight=lw["intensity"],
        confidence_weight=lw["confidence"],
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    if resume_path:
        start_epoch = load_checkpoint(Path(resume_path), system, optimizer, device)

    # ── Training loop ──────────────────────────────────────────────────────────
    ckpt_dir = Path(t_cfg["checkpoint_dir"])
    best_val_loss = float("inf")
    patience_counter = 0
    accumulator = MetricsAccumulator()

    for epoch in range(start_epoch, t_cfg["num_epochs"]):
        t0 = time.time()

        train_metrics = train_epoch(
            sat_wrapper, system, train_loader, optimizer, loss_fn, device,
            grad_clip=t_cfg["grad_clip"], accumulator=accumulator,
        )
        val_metrics = eval_epoch(
            sat_wrapper, system, val_loader, loss_fn, device, accumulator
        )

        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            "Epoch %3d/%d | %.1fs | "
            "Train loss: %.4f | Val loss: %.4f | "
            "Track err (mean): %.1f km | Wind MAE: %.1f kt | Pressure MAE: %.1f hPa",
            epoch + 1, t_cfg["num_epochs"], elapsed,
            train_metrics.get("mean_loss", 0),
            val_metrics.get("mean_loss", 0),
            val_metrics.get("track_error_mean", 0),
            val_metrics.get("wind_mae_mean", 0),
            val_metrics.get("pressure_mae_mean", 0),
        )

        # Save best checkpoint
        val_loss = val_metrics.get("mean_loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(ckpt_dir / "best_fusion.pt", system, optimizer, epoch, val_metrics)
        else:
            patience_counter += 1
            logger.info("No improvement (%d/%d).", patience_counter, t_cfg["patience"])

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                ckpt_dir / f"fusion_epoch_{epoch+1:03d}.pt",
                system, optimizer, epoch, val_metrics,
            )

        # Early stopping
        if patience_counter >= t_cfg["patience"]:
            logger.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal fusion system.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.resume)
