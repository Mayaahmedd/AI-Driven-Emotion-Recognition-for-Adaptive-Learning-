
import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.append("/kaggle/working/FER_Project/src_exp38")

from dataset_exp38 import DAiSEEMultilabelDataset, LABEL_COLS
from model_exp38 import ResNet50SEBiLSTM2AttentionMultilabel


BASE_INPUT = "/kaggle/input/datasets/mayaahmedd59/daisee-farmes"
FRAMES_ROOT = f"{BASE_INPUT}/frames/frames"

TRAIN_CSV = "/kaggle/working/FER_Project/processed_exp38/train_multilabel_exp38_oversampled.csv"
VAL_CSV = f"{BASE_INPUT}/val_multilabel.csv"

CHECKPOINT_DIR = "/kaggle/working/FER_Project/checkpoints/exp38_multilabel_resnet_se_bilstm2_attention_24frames"
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_FRAMES = 24
IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 12
LR = 5e-5
NUM_WORKERS = 2


def compute_pos_weight(csv_path):
    df = pd.read_csv(csv_path)
    positives = df[LABEL_COLS].sum().values.astype(np.float32)
    negatives = len(df) - positives

    pos_weight = negatives / np.maximum(positives, 1.0)

    # soften extreme weights slightly
    pos_weight = np.clip(pos_weight, 0.5, 6.0)

    return torch.tensor(pos_weight, dtype=torch.float32)


@torch.no_grad()
def collect_probs_labels(model, loader):
    model.eval()

    all_probs = []
    all_labels = []

    for batch in tqdm(loader, desc="Validation", leave=False):
        x = batch["frames"].to(DEVICE)
        y = batch["labels"].to(DEVICE)

        logits = model(x)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_probs, all_labels


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
        all_labels,
        preds,
        target_names=LABEL_COLS,
        output_dict=True,
        zero_division=0
    )

    return {
        "exact_match_accuracy": float(exact_match),
        "per_label_accuracy": {LABEL_COLS[i]: float(label_acc[i]) for i in range(len(LABEL_COLS))},
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "samples_f1": float(samples_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "thresholds": {LABEL_COLS[i]: float(thresholds[0, i]) for i in range(len(LABEL_COLS))}
    }


def tune_thresholds(all_probs, all_labels):
    best_thresholds = []

    grid = np.arange(0.10, 0.71, 0.05)

    for i, label in enumerate(LABEL_COLS):
        best_t = 0.5
        best_f1 = -1

        y_true = all_labels[:, i]

        for t in grid:
            y_pred = (all_probs[:, i] >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        best_thresholds.append(best_t)

    return best_thresholds


def main():
    print("Using device:", DEVICE)
    print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
    print("VAL_CSV exists:", os.path.exists(VAL_CSV))

    train_df = pd.read_csv(TRAIN_CSV)
    print("\nTraining positive counts:")
    print(train_df[LABEL_COLS].sum())

    pos_weight = compute_pos_weight(TRAIN_CSV).to(DEVICE)
    print("\npos_weight:")
    for name, weight in zip(LABEL_COLS, pos_weight.detach().cpu().numpy()):
        print(name, weight)

    train_ds = DAiSEEMultilabelDataset(
        TRAIN_CSV,
        FRAMES_ROOT,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        train=True
    )

    val_ds = DAiSEEMultilabelDataset(
        VAL_CSV,
        FRAMES_ROOT,
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = ResNet50SEBiLSTM2AttentionMultilabel(
        num_labels=4,
        pretrained=True,
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
        freeze_backbone=False
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_macro_f1 = -1
    logs = []
    patience = 4
    patience_counter = 0

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

        avg_loss = total_loss / len(train_loader)

        val_probs, val_labels = collect_probs_labels(model, val_loader)

        tuned_thresholds = tune_thresholds(val_probs, val_labels)
        val_metrics = metrics_from_probs(val_probs, val_labels, tuned_thresholds)

        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "macro_f1": val_metrics["macro_f1"],
            "micro_f1": val_metrics["micro_f1"],
            "samples_f1": val_metrics["samples_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "exact_match_accuracy": val_metrics["exact_match_accuracy"],
            "thresholds": val_metrics["thresholds"],
            "per_label_accuracy": val_metrics["per_label_accuracy"]
        }

        logs.append(row)

        print(f"\nEpoch {epoch}")
        print("Train Loss:", avg_loss)
        print("Val Macro F1:", val_metrics["macro_f1"])
        print("Val Micro F1:", val_metrics["micro_f1"])
        print("Val Samples F1:", val_metrics["samples_f1"])
        print("Val Exact Match Accuracy:", val_metrics["exact_match_accuracy"])
        print("Val Thresholds:", val_metrics["thresholds"])
        print("Val Per-label Accuracy:", val_metrics["per_label_accuracy"])

        with open(f"{CHECKPOINT_DIR}/train_log.json", "w") as f:
            json.dump(logs, f, indent=2)

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
                    "thresholds": tuned_thresholds,
                    "pos_weight": pos_weight.detach().cpu(),
                    "model": "Exp38_Multilabel_ResNet50_SE_2LayerBiLSTM_TemporalAttention_24Frames",
                    "note": "Exp38 enhances Exp34 using 24 frames, SE attention, 2-layer BiLSTM, pos_weight, augmentation, and threshold tuning."
                },
                ckpt_path
            )

            with open(f"{CHECKPOINT_DIR}/best_val_report.json", "w") as f:
                json.dump(val_metrics, f, indent=2)

            print("Saved best checkpoint:", ckpt_path)

        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete. Best Macro F1:", best_macro_f1)


if __name__ == "__main__":
    main()
