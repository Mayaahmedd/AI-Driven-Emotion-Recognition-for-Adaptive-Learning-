"""
Training script for exp41 (stability-enhanced).

Differences vs the exp38/exp40-C baseline
----------------------------------------
1. Exponential Moving Average of model weights with decay = 0.999. The
   EMA module is updated after every ``optimizer.step()``. Validation
   metrics are reported for BOTH the live model and the EMA model on
   every epoch, so the EMA vs non-EMA comparison can be read off the
   training log without re-running anything.
2. Cosine annealing learning-rate schedule with a 1-epoch linear
   warm-up, stepped per iteration. Replaces the exp38/exp40-C
   ``ReduceLROnPlateau`` schedule.
3. The top-3 best EMA checkpoints (by validation Macro F1) are tracked
   throughout training so that the evaluation script can optionally
   average their state dicts at test time.

Held constant (no confounds)
----------------------------
- Architecture: ResNet50 + SE + 2-layer BiLSTM + Temporal Attention.
- Data pipeline: per-frame augmentation policy, T = 24, 224x224,
  ImageNet normalisation.
- Training set: dominant-group oversampled CSV (same rule as exp38 /
  exp40-C).
- Loss: ``BCEWithLogitsLoss`` with ``pos_weight`` clipped to [0.5, 6.0],
  computed from the oversampled CSV.
- Threshold tuning: per-label grid search 0.10 -> 0.70, step 0.05, on
  validation, applied each epoch.
- Optimiser: AdamW, lr = 5e-5, weight decay = 1e-5.
- Batch size 4, grad-clip 5.0, 12 epochs, early-stopping patience 4
  (on EMA validation Macro F1).
- Seed 42 with ``set_all_seeds`` and ``worker_init_fn``.

Forbidden by design
-------------------
Mixup, label smoothing, architecture changes, new losses, new
augmentation policies. This script will refuse to enable any of them.
"""

from __future__ import annotations

import json
import os
import random
import sys
from heapq import heappush, heappushpop
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/kaggle/working/FER_Project/src_exp41")

from dataset_exp41 import DAiSEEMultilabelDatasetExp41, LABEL_COLS
from ema import ModelEMA
from model_exp41 import ResNet50SEBiLSTM2AttentionMultilabelExp41
from scheduler import build_cosine_with_warmup


# ============================================================================
# Configuration (held fixed vs exp38 / exp40-C baseline)
# ============================================================================

BASE_INPUT = "/kaggle/input/datasets/mayaahmedd59/daisee-farmes"
FRAMES_ROOT = f"{BASE_INPUT}/frames/frames"

TRAIN_CSV = "/kaggle/working/FER_Project/processed_exp41/train_multilabel_exp41_oversampled.csv"
VAL_CSV = f"{BASE_INPUT}/val_multilabel.csv"

CHECKPOINT_DIR = Path("/kaggle/working/FER_Project/checkpoints/exp41_stability_enhanced")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_FRAMES = 24
IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 12
LR = 5e-5
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2
SEED = 42

# Stability-only interventions
EMA_DECAY = 0.999
WARMUP_EPOCHS = 1            # 1-epoch linear warm-up before cosine
COSINE_ETA_MIN_RATIO = 0.0   # cosine down to zero by the end of training
TOP_K_FOR_ENSEMBLE = 3       # number of best EMA checkpoints retained

# Imbalance handling (preserved from exp38 / exp40-C)
POS_WEIGHT_CLIP = (0.5, 6.0)
THRESHOLD_GRID = np.arange(0.10, 0.71, 0.05)

# Other training-loop constants
EARLY_STOP_PATIENCE = 4
GRAD_CLIP_NORM = 5.0


# ============================================================================
# Reproducibility utilities
# ============================================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int) -> None:
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# ============================================================================
# Imbalance: per-label pos_weight (clipped, computed from oversampled CSV)
# ============================================================================

def compute_pos_weight(csv_path: str) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    pos = df[LABEL_COLS].sum().values.astype(np.float32)
    neg = len(df) - pos
    w = neg / np.maximum(pos, 1.0)
    w = np.clip(w, POS_WEIGHT_CLIP[0], POS_WEIGHT_CLIP[1])
    return torch.tensor(w, dtype=torch.float32)


# ============================================================================
# Validation utilities
# ============================================================================

@torch.no_grad()
def collect_probs_labels(model: nn.Module, loader) -> tuple:
    was_training = model.training
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="Validation", leave=False):
        x = batch["frames"].to(DEVICE)
        y = batch["labels"].to(DEVICE)
        probs = torch.sigmoid(model(x))
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(all_probs, 0), np.concatenate(all_labels, 0)


def tune_thresholds(probs: np.ndarray, labels: np.ndarray):
    best = []
    for i in range(labels.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in THRESHOLD_GRID:
            pred = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        best.append(best_t)
    return best


def metrics_from_probs(probs, labels, thresholds):
    thr = np.array(thresholds).reshape(1, -1)
    preds = (probs >= thr).astype(int)
    label_acc = (preds == labels).mean(axis=0)
    return {
        "exact_match_accuracy": float(accuracy_score(labels, preds)),
        "per_label_accuracy": {LABEL_COLS[i]: float(label_acc[i]) for i in range(len(LABEL_COLS))},
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(labels, preds, average="samples", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            labels, preds, target_names=LABEL_COLS, output_dict=True, zero_division=0
        ),
        "thresholds": {LABEL_COLS[i]: float(thr[0, i]) for i in range(len(LABEL_COLS))},
    }


# ============================================================================
# Main
# ============================================================================

def main():
    set_all_seeds(SEED)

    print("=" * 80)
    print("EXP41 — STABILITY-ENHANCED TRAINING")
    print(f"  EMA decay        : {EMA_DECAY}")
    print(f"  LR schedule      : cosine annealing with {WARMUP_EPOCHS}-epoch warmup")
    print(f"  Top-k tracked    : {TOP_K_FOR_ENSEMBLE}")
    print("  Architecture     : ResNet50 + SE + 2-layer BiLSTM + Temporal Attention (UNCHANGED)")
    print("  Imbalance        : oversampling + pos_weight (same as exp40-C)")
    print("=" * 80)

    print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
    print("VAL_CSV exists  :", os.path.exists(VAL_CSV))

    train_df = pd.read_csv(TRAIN_CSV)
    print("\nTraining positive counts:")
    print(train_df[LABEL_COLS].sum())

    pos_weight = compute_pos_weight(TRAIN_CSV).to(DEVICE)
    print("\npos_weight:", {LABEL_COLS[i]: float(pos_weight[i]) for i in range(len(LABEL_COLS))})

    # ---- datasets + loaders ------------------------------------------------
    train_ds = DAiSEEMultilabelDatasetExp41(
        TRAIN_CSV, FRAMES_ROOT,
        num_frames=NUM_FRAMES, image_size=IMAGE_SIZE, train=True,
    )
    val_ds = DAiSEEMultilabelDatasetExp41(
        VAL_CSV, FRAMES_ROOT,
        num_frames=NUM_FRAMES, image_size=IMAGE_SIZE, train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # ---- model + EMA + optimiser + scheduler -------------------------------
    model = ResNet50SEBiLSTM2AttentionMultilabelExp41(
        num_labels=4, pretrained=True,
        hidden_size=256, num_layers=2,
        dropout=0.5, freeze_backbone=False,
    ).to(DEVICE)

    ema = ModelEMA(model, decay=EMA_DECAY).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    iters_per_epoch = max(len(train_loader), 1)
    total_iters = EPOCHS * iters_per_epoch
    warmup_iters = WARMUP_EPOCHS * iters_per_epoch
    scheduler = build_cosine_with_warmup(
        optimizer,
        total_iters=total_iters,
        warmup_iters=warmup_iters,
        eta_min_ratio=COSINE_ETA_MIN_RATIO,
    )
    print(f"\nScheduler: total_iters={total_iters}, warmup_iters={warmup_iters}, eta_min_ratio={COSINE_ETA_MIN_RATIO}")

    # ---- top-k EMA checkpoint tracking -------------------------------------
    # Min-heap of (macro_f1_ema, epoch, ckpt_path). The path on disk for the
    # k-th best is recreated each time the heap is updated so that all top-k
    # checkpoints remain available for the optional ensemble.
    topk_heap = []  # type: ignore[var-annotated]

    def topk_path(rank: int) -> Path:
        return CHECKPOINT_DIR / f"best_ema_top{rank}.pt"

    def maybe_save_topk(score_ema: float, epoch: int, ema_state: dict, thresholds: list) -> None:
        """Insert candidate into the top-k heap and persist files in rank order."""
        record_payload = {
            "ema_state_dict": ema_state,
            "epoch": epoch,
            "macro_f1_ema": float(score_ema),
            "thresholds": thresholds,
            "num_frames": NUM_FRAMES,
            "image_size": IMAGE_SIZE,
            "label_cols": LABEL_COLS,
            "pos_weight": pos_weight.detach().cpu(),
            "ema_decay": EMA_DECAY,
        }
        if len(topk_heap) < TOP_K_FOR_ENSEMBLE:
            heappush(topk_heap, (float(score_ema), epoch, record_payload))
        else:
            if float(score_ema) > topk_heap[0][0]:
                heappushpop(topk_heap, (float(score_ema), epoch, record_payload))
        # Re-write the on-disk top-k snapshots in rank order (best -> worst).
        ranked = sorted(topk_heap, key=lambda r: (-r[0], -r[1]))
        for i, rec in enumerate(ranked, start=1):
            torch.save(rec[2], topk_path(i))

    # ---- training loop -----------------------------------------------------
    best_macro_f1_ema = -1.0
    best_macro_f1_live = -1.0
    patience_counter = 0
    logs = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            x = batch["frames"].to(DEVICE)
            y = batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        avg_loss = total_loss / max(len(train_loader), 1)

        # ---- evaluate both live and EMA on validation ----------------------
        probs_live, labels_val = collect_probs_labels(model, val_loader)
        thr_live = tune_thresholds(probs_live, labels_val)
        val_live = metrics_from_probs(probs_live, labels_val, thr_live)

        probs_ema, _ = collect_probs_labels(ema.module, val_loader)
        thr_ema = tune_thresholds(probs_ema, labels_val)
        val_ema = metrics_from_probs(probs_ema, labels_val, thr_ema)

        row = {
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "lr_end_of_epoch": float(optimizer.param_groups[0]["lr"]),
            "live": {
                "macro_f1": val_live["macro_f1"],
                "micro_f1": val_live["micro_f1"],
                "samples_f1": val_live["samples_f1"],
                "weighted_f1": val_live["weighted_f1"],
                "exact_match_accuracy": val_live["exact_match_accuracy"],
                "thresholds": val_live["thresholds"],
            },
            "ema": {
                "macro_f1": val_ema["macro_f1"],
                "micro_f1": val_ema["micro_f1"],
                "samples_f1": val_ema["samples_f1"],
                "weighted_f1": val_ema["weighted_f1"],
                "exact_match_accuracy": val_ema["exact_match_accuracy"],
                "thresholds": val_ema["thresholds"],
            },
            "ema_minus_live_macro_f1": val_ema["macro_f1"] - val_live["macro_f1"],
        }
        logs.append(row)

        print(f"\nEpoch {epoch} | lr_end={row['lr_end_of_epoch']:.2e}")
        print(f"  train loss        : {avg_loss:.4f}")
        print(f"  val Macro F1 LIVE : {val_live['macro_f1']:.4f}")
        print(f"  val Macro F1 EMA  : {val_ema['macro_f1']:.4f}   (Δ = {row['ema_minus_live_macro_f1']:+.4f})")
        print(f"  val thresholds EMA: {val_ema['thresholds']}")

        (CHECKPOINT_DIR / "train_log.json").write_text(json.dumps(logs, indent=2))

        # ---- track best LIVE checkpoint for completeness -------------------
        if val_live["macro_f1"] > best_macro_f1_live:
            best_macro_f1_live = val_live["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_macro_f1_live": best_macro_f1_live,
                    "thresholds": thr_live,
                    "num_frames": NUM_FRAMES,
                    "image_size": IMAGE_SIZE,
                    "label_cols": LABEL_COLS,
                    "pos_weight": pos_weight.detach().cpu(),
                    "ema_decay": EMA_DECAY,
                },
                CHECKPOINT_DIR / "best_model_live.pt",
            )
            (CHECKPOINT_DIR / "best_val_report_live.json").write_text(json.dumps(val_live, indent=2))

        # ---- track best EMA checkpoint + top-k -----------------------------
        improved_ema = val_ema["macro_f1"] > best_macro_f1_ema
        if improved_ema:
            best_macro_f1_ema = val_ema["macro_f1"]
            patience_counter = 0
            torch.save(
                {
                    "ema_state_dict": ema.state_dict(),
                    "epoch": epoch,
                    "best_macro_f1_ema": best_macro_f1_ema,
                    "thresholds": thr_ema,
                    "num_frames": NUM_FRAMES,
                    "image_size": IMAGE_SIZE,
                    "label_cols": LABEL_COLS,
                    "pos_weight": pos_weight.detach().cpu(),
                    "ema_decay": EMA_DECAY,
                },
                CHECKPOINT_DIR / "best_model_ema.pt",
            )
            (CHECKPOINT_DIR / "best_val_report_ema.json").write_text(json.dumps(val_ema, indent=2))
            print(f"  saved best EMA checkpoint -> {CHECKPOINT_DIR / 'best_model_ema.pt'}")
        else:
            patience_counter += 1
            print(f"  no EMA improvement (patience {patience_counter}/{EARLY_STOP_PATIENCE})")

        # Always offer the current EMA snapshot to the top-k heap so the
        # ensemble can include high-quality non-best epochs.
        maybe_save_topk(val_ema["macro_f1"], epoch, ema.state_dict(), thr_ema)

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

    # ---- finalise top-k summary -------------------------------------------
    ranked = sorted(topk_heap, key=lambda r: (-r[0], -r[1]))
    topk_summary = [
        {"rank": i + 1, "macro_f1_ema": float(rec[0]), "epoch": int(rec[1])}
        for i, rec in enumerate(ranked)
    ]
    (CHECKPOINT_DIR / "topk_summary.json").write_text(json.dumps(topk_summary, indent=2))

    print("\nTraining complete.")
    print(f"  best val Macro F1 LIVE : {best_macro_f1_live:.4f}")
    print(f"  best val Macro F1 EMA  : {best_macro_f1_ema:.4f}")
    print(f"  top-{TOP_K_FOR_ENSEMBLE} EMA snapshots:")
    for rec in topk_summary:
        print(f"    rank {rec['rank']}: epoch {rec['epoch']} | Macro F1 = {rec['macro_f1_ema']:.4f}")


if __name__ == "__main__":
    main()
