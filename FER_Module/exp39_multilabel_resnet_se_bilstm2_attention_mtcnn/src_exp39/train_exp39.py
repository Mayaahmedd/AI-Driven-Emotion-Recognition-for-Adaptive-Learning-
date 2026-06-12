"""
Training script for exp39 (per-clip augmentation + MTCNN face crop).

Differences vs exp38 train script
---------------------------------
1. Full deterministic seeding of Python, NumPy, PyTorch, CUDA, the
   DataLoader workers and the in-dataset augmentation RNG.
2. MTCNN face detector instantiated once on the CPU side and shared across
   workers via a bounding-box cache file written to
   ``processed_exp39/face_bboxes.json``. The first epoch warms the cache;
   subsequent epochs reuse it.
3. The validation-set protocol is unchanged from exp38: per-label
   thresholds are retuned on the validation set after every epoch and the
   thresholds attached to the best-Macro-F1 checkpoint are the ones used
   later by ``evaluate_exp39.py`` on the held-out test set. The test set is
   never touched during training.
4. Hyperparameters (lr 5e-5, weight decay 1e-5, AdamW, ReduceLROnPlateau
   factor 0.5 patience 2, 12 epochs, early stop patience 4, BCEWithLogitsLoss
   with pos_weight clipped to [0.5, 6.0], batch size 4, grad-clip 5.0) are
   preserved exactly so any quantitative gap between exp38 and exp39 is
   attributable to the data-pipeline changes only.

This file is intended to be uploaded to Kaggle and executed from
``/kaggle/working/FER_Project/src_exp39``.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/kaggle/working/FER_Project/src_exp39")

from dataset_exp39 import DAiSEEMultilabelDatasetExp39, LABEL_COLS
from face_detector_exp39 import MTCNNFaceDetector
from model_exp39 import ResNet50SEBiLSTM2AttentionMultilabelExp39


# ============================================================================
# Configuration
# ============================================================================

BASE_INPUT = "/kaggle/input/datasets/mayaahmedd59/daisee-farmes"
FRAMES_ROOT = f"{BASE_INPUT}/frames/frames"

TRAIN_CSV = "/kaggle/working/FER_Project/processed_exp39/train_multilabel_exp39_oversampled.csv"
VAL_CSV = f"{BASE_INPUT}/val_multilabel.csv"

CHECKPOINT_DIR = "/kaggle/working/FER_Project/checkpoints/exp39_multilabel_resnet_se_bilstm2_attention_mtcnn"
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

BBOX_CACHE_PATH = "/kaggle/working/FER_Project/processed_exp39/face_bboxes.json"
Path(BBOX_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_FRAMES = 24
IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 12
LR = 5e-5
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2
SEED = 42
FACE_MARGIN = 0.20
USE_FACE_DETECTION = True


# ============================================================================
# Reproducibility utilities
# ============================================================================

def set_all_seeds(seed: int) -> None:
    """Seed every RNG that can affect training.

    Notes
    -----
    * cuDNN benchmarking is disabled to make convolution algorithms
      deterministic at a (mild) throughput cost.
    * ``torch.use_deterministic_algorithms`` is *not* enabled because it
      requires ``CUBLAS_WORKSPACE_CONFIG`` to be set and conflicts with
      ``bidirectional`` LSTM kernels on some CUDA versions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int) -> None:
    """Per-worker seed callback for DataLoader."""
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# ============================================================================
# Imbalance: per-label positive-class weight (preserved from exp38)
# ============================================================================

def compute_pos_weight(csv_path: str) -> torch.Tensor:
    df = pd.read_csv(csv_path)
    positives = df[LABEL_COLS].sum().values.astype(np.float32)
    negatives = len(df) - positives
    pos_weight = negatives / np.maximum(positives, 1.0)
    pos_weight = np.clip(pos_weight, 0.5, 6.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


# ============================================================================
# Validation utilities (identical metric set to exp38)
# ============================================================================

@torch.no_grad()
def collect_probs_labels(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="Validation", leave=False):
        x = batch["frames"].to(DEVICE)
        y = batch["labels"].to(DEVICE)
        logits = model(x)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def metrics_from_probs(all_probs, all_labels, thresholds):
    thresholds = np.array(thresholds).reshape(1, -1)
    preds = (all_probs >= thresholds).astype(int)
    label_acc = (preds == all_labels).mean(axis=0)
    exact_match = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    samples_f1 = f1_score(all_labels, preds, average="samples", zero_division=0)
    weighted_f1 = f1_score(all_labels, preds, average="weighted", zero_division=0)
    report = classification_report(
        all_labels, preds, target_names=LABEL_COLS, output_dict=True, zero_division=0
    )
    return {
        "exact_match_accuracy": float(exact_match),
        "per_label_accuracy": {LABEL_COLS[i]: float(label_acc[i]) for i in range(len(LABEL_COLS))},
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "samples_f1": float(samples_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "thresholds": {LABEL_COLS[i]: float(thresholds[0, i]) for i in range(len(LABEL_COLS))},
    }


def tune_thresholds(all_probs, all_labels):
    best_thresholds = []
    grid = np.arange(0.10, 0.71, 0.05)
    for i in range(all_labels.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        y_true = all_labels[:, i]
        for t in grid:
            y_pred = (all_probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        best_thresholds.append(best_t)
    return best_thresholds


# ============================================================================
# Main
# ============================================================================

def main():
    set_all_seeds(SEED)

    print("Using device:", DEVICE)
    print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
    print("VAL_CSV exists:", os.path.exists(VAL_CSV))

    train_df = pd.read_csv(TRAIN_CSV)
    print("\nTraining positive counts:")
    print(train_df[LABEL_COLS].sum())

    pos_weight = compute_pos_weight(TRAIN_CSV).to(DEVICE)
    print("\npos_weight:")
    for name, w in zip(LABEL_COLS, pos_weight.detach().cpu().numpy()):
        print(name, w)

    # ---- shared face detector with persistent disk cache ------------------
    face_detector = (
        MTCNNFaceDetector(
            device="cuda" if torch.cuda.is_available() else "cpu",
            min_face_size=40,
            margin=FACE_MARGIN,
            cache_path=BBOX_CACHE_PATH,
        )
        if USE_FACE_DETECTION
        else None
    )

    # ---- datasets ---------------------------------------------------------
    train_ds = DAiSEEMultilabelDatasetExp39(
        TRAIN_CSV, FRAMES_ROOT,
        num_frames=NUM_FRAMES, image_size=IMAGE_SIZE,
        train=True, face_detector=face_detector, seed=SEED,
    )
    val_ds = DAiSEEMultilabelDatasetExp39(
        VAL_CSV, FRAMES_ROOT,
        num_frames=NUM_FRAMES, image_size=IMAGE_SIZE,
        train=False, face_detector=face_detector, seed=SEED,
    )

    # ---- loaders ----------------------------------------------------------
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

    # ---- model ------------------------------------------------------------
    model = ResNet50SEBiLSTM2AttentionMultilabelExp39(
        num_labels=4, pretrained=True,
        hidden_size=256, num_layers=2,
        dropout=0.5, freeze_backbone=False,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    best_macro_f1 = -1.0
    logs = []
    patience, patience_counter = 4, 0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / max(len(train_loader), 1)

        # persist MTCNN cache after every epoch so a crashed run can resume
        if face_detector is not None:
            face_detector.save_cache()

        val_probs, val_labels = collect_probs_labels(model, val_loader)
        tuned = tune_thresholds(val_probs, val_labels)
        val_metrics = metrics_from_probs(val_probs, val_labels, tuned)

        scheduler.step(val_metrics["macro_f1"])

        logs.append({
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "macro_f1": val_metrics["macro_f1"],
            "micro_f1": val_metrics["micro_f1"],
            "samples_f1": val_metrics["samples_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "exact_match_accuracy": val_metrics["exact_match_accuracy"],
            "thresholds": val_metrics["thresholds"],
            "per_label_accuracy": val_metrics["per_label_accuracy"],
        })

        print(f"\nEpoch {epoch}")
        print("Train Loss:", avg_loss)
        print("Val Macro F1:", val_metrics["macro_f1"])
        print("Val Micro F1:", val_metrics["micro_f1"])
        print("Val Samples F1:", val_metrics["samples_f1"])
        print("Val Exact Match Accuracy:", val_metrics["exact_match_accuracy"])
        print("Val Thresholds:", val_metrics["thresholds"])
        print("Val Per-label Accuracy:", val_metrics["per_label_accuracy"])

        with open(f"{CHECKPOINT_DIR}/train_log.json", "w") as fh:
            json.dump(logs, fh, indent=2)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            ckpt_path = f"{CHECKPOINT_DIR}/best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_macro_f1": best_macro_f1,
                    "num_frames": NUM_FRAMES,
                    "image_size": IMAGE_SIZE,
                    "label_cols": LABEL_COLS,
                    "thresholds": tuned,
                    "pos_weight": pos_weight.detach().cpu(),
                    "seed": SEED,
                    "face_margin": FACE_MARGIN,
                    "use_face_detection": USE_FACE_DETECTION,
                    "model": "Exp39_Multilabel_ResNet50_SE_2LayerBiLSTM_TemporalAttention_24Frames_MTCNN",
                    "note": (
                        "Exp39 replaces per-frame augmentation with per-clip "
                        "augmentation and adds MTCNN face cropping with margin "
                        f"{FACE_MARGIN}; all other hyperparameters match exp38."
                    ),
                },
                ckpt_path,
            )
            with open(f"{CHECKPOINT_DIR}/best_val_report.json", "w") as fh:
                json.dump(val_metrics, fh, indent=2)
            print("Saved best checkpoint:", ckpt_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if face_detector is not None:
        face_detector.save_cache()

    print("Training complete. Best Macro F1:", best_macro_f1)


if __name__ == "__main__":
    main()
