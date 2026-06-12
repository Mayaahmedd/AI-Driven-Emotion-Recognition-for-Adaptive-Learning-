"""
Training script for one variant of the exp40 imbalance ablation.

Usage
-----
    python train_exp40.py --config configs/variant_A_oversampling_only.json
    python train_exp40.py --config configs/variant_B_posweight_only.json
    python train_exp40.py --config configs/variant_C_combined.json

The script reads two binary switches from the config file:

    use_oversampling   ─ if True, train on the dominant-group oversampled
                         CSV; if False, train on the original DAiSEE
                         train_multilabel.csv.
    use_pos_weight     ─ if True, build BCE pos_weight from the variant's
                         training CSV (clipped to [0.5, 6.0]); if False,
                         pos_weight is ones(4) so BCE is unweighted.

Everything else (architecture, optimiser, scheduler, augmentation, T=24
frame sampling, batch size, threshold-tuning grid, early-stopping rule,
metric set) is held constant across A, B and C so that the ablation
isolates only the imbalance-correction strategy.
"""

from __future__ import annotations

import argparse
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

sys.path.append("/kaggle/working/FER_Project/src_exp40")

from dataset_exp40 import DAiSEEMultilabelDatasetExp40, LABEL_COLS
from model_exp40 import ResNet50SEBiLSTM2AttentionMultilabelExp40


# ============================================================================
# Constants held fixed across the three variants
# ============================================================================

BASE_INPUT = "/kaggle/input/datasets/mayaahmedd59/daisee-farmes"
FRAMES_ROOT = f"{BASE_INPUT}/frames/frames"

ORIGINAL_TRAIN_CSV = f"{BASE_INPUT}/train_multilabel.csv"
OVERSAMPLED_TRAIN_CSV = "/kaggle/working/FER_Project/processed_exp40/train_multilabel_exp40_oversampled.csv"
VAL_CSV = f"{BASE_INPUT}/val_multilabel.csv"

RESULTS_ROOT = Path("/kaggle/working/FER_Project/checkpoints/exp40_imbalance_ablation")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_FRAMES = 24
IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 12
LR = 5e-5
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2
SEED = 42
POS_WEIGHT_CLIP = (0.5, 6.0)
THRESHOLD_GRID = np.arange(0.10, 0.71, 0.05)
EARLY_STOP_PATIENCE = 4
SCHEDULER_PATIENCE = 2
GRAD_CLIP_NORM = 5.0


# ============================================================================
# Helpers
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


def select_training_csv(use_oversampling: bool) -> str:
    return OVERSAMPLED_TRAIN_CSV if use_oversampling else ORIGINAL_TRAIN_CSV


def build_pos_weight(train_csv: str, use_pos_weight: bool) -> torch.Tensor:
    if not use_pos_weight:
        return torch.ones(len(LABEL_COLS), dtype=torch.float32)
    df = pd.read_csv(train_csv)
    pos = df[LABEL_COLS].sum().values.astype(np.float32)
    neg = len(df) - pos
    w = neg / np.maximum(pos, 1.0)
    w = np.clip(w, POS_WEIGHT_CLIP[0], POS_WEIGHT_CLIP[1])
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def collect_probs_labels(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in tqdm(loader, desc="Validation", leave=False):
        x = batch["frames"].to(DEVICE)
        y = batch["labels"].to(DEVICE)
        probs = torch.sigmoid(model(x))
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return np.concatenate(all_probs, 0), np.concatenate(all_labels, 0)


def tune_thresholds(probs, labels):
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


def per_label_confusion(labels, preds):
    """For each label compute TP/FP/TN/FN plus precision, recall, F1, accuracy."""
    out = {}
    for i, name in enumerate(LABEL_COLS):
        y_true = labels[:, i].astype(int)
        y_pred = preds[:, i].astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        acc = (tp + tn) / max(tp + fp + tn + fn, 1)
        out[name] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "support_positive": tp + fn,
            "support_negative": tn + fp,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "accuracy": float(acc),
        }
    return out


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
        "per_label_confusion": per_label_confusion(labels, preds),
        "thresholds": {LABEL_COLS[i]: float(thr[0, i]) for i in range(len(LABEL_COLS))},
    }


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="Path to a variant config JSON in configs/.")
    args = ap.parse_args()

    with open(args.config, "r") as fh:
        cfg = json.load(fh)

    variant = cfg["variant"]
    name = cfg["name"]
    use_oversampling = bool(cfg["use_oversampling"])
    use_pos_weight = bool(cfg["use_pos_weight"])

    set_all_seeds(SEED)

    print("=" * 80)
    print(f"EXP40 IMBALANCE ABLATION — VARIANT {variant} ({name})")
    print(f"use_oversampling: {use_oversampling}")
    print(f"use_pos_weight  : {use_pos_weight}")
    print(f"description     : {cfg.get('description', '')}")
    print("=" * 80)

    train_csv = select_training_csv(use_oversampling)
    print("Train CSV:", train_csv)
    print("Train CSV exists:", os.path.exists(train_csv))
    print("Val CSV:", VAL_CSV)
    print("Val CSV exists:", os.path.exists(VAL_CSV))

    if use_oversampling and not os.path.exists(OVERSAMPLED_TRAIN_CSV):
        raise FileNotFoundError(
            "Oversampled CSV not found. Run make_balanced_csv_exp40.py first."
        )

    pos_weight = build_pos_weight(train_csv, use_pos_weight).to(DEVICE)
    print("pos_weight:", {LABEL_COLS[i]: float(pos_weight[i]) for i in range(len(LABEL_COLS))})

    train_ds = DAiSEEMultilabelDatasetExp40(
        train_csv, FRAMES_ROOT,
        num_frames=NUM_FRAMES, image_size=IMAGE_SIZE, train=True,
    )
    val_ds = DAiSEEMultilabelDatasetExp40(
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

    model = ResNet50SEBiLSTM2AttentionMultilabelExp40(
        num_labels=4, pretrained=True,
        hidden_size=256, num_layers=2,
        dropout=0.5, freeze_backbone=False,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE,
    )

    variant_dir = RESULTS_ROOT / f"variant_{variant}_{name}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    best_macro_f1 = -1.0
    logs = []
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Variant {variant}] Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            x = batch["frames"].to(DEVICE)
            y = batch["labels"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / max(len(train_loader), 1)

        probs, labels = collect_probs_labels(model, val_loader)
        thresholds = tune_thresholds(probs, labels)
        val_metrics = metrics_from_probs(probs, labels, thresholds)

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

        print(f"\n[Variant {variant}] Epoch {epoch}")
        print(f"  train loss : {avg_loss:.4f}")
        print(f"  macro F1   : {val_metrics['macro_f1']:.4f}")
        print(f"  micro F1   : {val_metrics['micro_f1']:.4f}")
        print(f"  samples F1 : {val_metrics['samples_f1']:.4f}")
        print(f"  exact-match: {val_metrics['exact_match_accuracy']:.4f}")
        print(f"  thresholds : {val_metrics['thresholds']}")

        (variant_dir / "train_log.json").write_text(json.dumps(logs, indent=2))

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_macro_f1": best_macro_f1,
                    "num_frames": NUM_FRAMES,
                    "image_size": IMAGE_SIZE,
                    "label_cols": LABEL_COLS,
                    "thresholds": thresholds,
                    "pos_weight": pos_weight.detach().cpu(),
                    "variant": variant,
                    "name": name,
                    "use_oversampling": use_oversampling,
                    "use_pos_weight": use_pos_weight,
                },
                variant_dir / "best_model.pt",
            )
            (variant_dir / "best_val_report.json").write_text(json.dumps(val_metrics, indent=2))
            print(f"  saved checkpoint -> {variant_dir / 'best_model.pt'}")
        else:
            patience_counter += 1
            print(f"  no improvement (patience {patience_counter}/{EARLY_STOP_PATIENCE})")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\n[Variant {variant}] training complete. Best val Macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
