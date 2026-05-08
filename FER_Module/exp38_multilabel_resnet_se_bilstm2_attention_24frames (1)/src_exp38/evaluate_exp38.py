
import sys
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.append("/kaggle/working/FER_Project/src_exp38")

from dataset_exp38 import DAiSEEMultilabelDataset, LABEL_COLS
from model_exp38 import ResNet50SEBiLSTM2AttentionMultilabel


BASE_INPUT = "/kaggle/input/datasets/mayaahmedd59/daisee-farmes"
FRAMES_ROOT = f"{BASE_INPUT}/frames/frames"
TEST_CSV = f"{BASE_INPUT}/test_multilabel.csv"

CHECKPOINT_DIR = "/kaggle/working/FER_Project/checkpoints/exp38_multilabel_resnet_se_bilstm2_attention_24frames"
CKPT_PATH = f"{CHECKPOINT_DIR}/best_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_WORKERS = 2


@torch.no_grad()
def main():
    print("Using device:", DEVICE)
    print("Checkpoint:", CKPT_PATH)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    num_frames = ckpt.get("num_frames", 24)
    image_size = ckpt.get("image_size", 224)
    thresholds = ckpt.get("thresholds", [0.5, 0.5, 0.5, 0.5])

    print("Using thresholds:", thresholds)

    test_ds = DAiSEEMultilabelDataset(
        TEST_CSV,
        FRAMES_ROOT,
        num_frames=num_frames,
        image_size=image_size,
        train=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = ResNet50SEBiLSTM2AttentionMultilabel(
        num_labels=4,
        pretrained=False,
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
        freeze_backbone=False
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_probs = []
    all_labels = []
    all_clip_ids = []

    for batch in tqdm(test_loader, desc="Testing"):
        x = batch["frames"].to(DEVICE)
        y = batch["labels"].to(DEVICE)

        logits = model(x)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_clip_ids.extend(batch["clip_id"])

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    thresholds_np = np.array(thresholds).reshape(1, -1)
    preds = (all_probs >= thresholds_np).astype(int)

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

    final_report = {
        "model": "Exp38 Multilabel ResNet50 + SE + 2-layer BiLSTM + Attention + 24 Frames",
        "thresholds": {LABEL_COLS[i]: float(thresholds[i]) for i in range(len(LABEL_COLS))},
        "exact_match_accuracy": float(exact_match),
        "per_label_accuracy": {LABEL_COLS[i]: float(label_acc[i]) for i in range(len(LABEL_COLS))},
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "samples_f1": float(samples_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "label_cols": LABEL_COLS
    }

    print("\n===== EXP38 MULTILABEL TEST RESULTS =====")
    print("Exact Match Accuracy:", exact_match)
    print("Per-label Accuracy:", final_report["per_label_accuracy"])
    print("Macro F1:", macro_f1)
    print("Micro F1:", micro_f1)
    print("Samples F1:", samples_f1)
    print("Weighted F1:", weighted_f1)
    print("Thresholds:", final_report["thresholds"])

    print("\nClassification report:")
    print(classification_report(all_labels, preds, target_names=LABEL_COLS, zero_division=0))

    with open(f"{CHECKPOINT_DIR}/test_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    with open(f"{CHECKPOINT_DIR}/test_report.txt", "w") as f:
        f.write("Exp38 Multilabel ResNet50 + SE + 2-layer BiLSTM + Attention + 24 Frames\n")
        f.write("=" * 80 + "\n")
        f.write(f"Exact Match Accuracy: {exact_match:.6f}\n")
        f.write(f"Macro F1: {macro_f1:.6f}\n")
        f.write(f"Micro F1: {micro_f1:.6f}\n")
        f.write(f"Samples F1: {samples_f1:.6f}\n")
        f.write(f"Weighted F1: {weighted_f1:.6f}\n")
        f.write(f"Thresholds: {final_report['thresholds']}\n\n")
        f.write(classification_report(all_labels, preds, target_names=LABEL_COLS, zero_division=0))

    pred_df = pd.DataFrame({
        "ClipID": all_clip_ids
    })

    for i, name in enumerate(LABEL_COLS):
        pred_df[f"true_{name}"] = all_labels[:, i].astype(int)
        pred_df[f"pred_{name}"] = preds[:, i].astype(int)
        pred_df[f"prob_{name}"] = all_probs[:, i]

    pred_df.to_csv(f"{CHECKPOINT_DIR}/test_predictions.csv", index=False)

    print("\nSaved:")
    print(f"{CHECKPOINT_DIR}/test_report.json")
    print(f"{CHECKPOINT_DIR}/test_report.txt")
    print(f"{CHECKPOINT_DIR}/test_predictions.csv")


if __name__ == "__main__":
    main()
