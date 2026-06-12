"""
Test-set evaluation for exp39.

The trained checkpoint carries the per-label thresholds that were tuned on
the validation set during training; we reuse those thresholds verbatim on
the held-out DAiSEE test set. The test set is never used to retune any
hyperparameter or threshold, which keeps the evaluation protocol identical
to exp38 and to mainstream DAiSEE practice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/kaggle/working/FER_Project/src_exp39")

from dataset_exp39 import DAiSEEMultilabelDatasetExp39, LABEL_COLS
from face_detector_exp39 import MTCNNFaceDetector
from model_exp39 import ResNet50SEBiLSTM2AttentionMultilabelExp39


BASE_INPUT = "/kaggle/input/datasets/mayaahmedd59/daisee-farmes"
FRAMES_ROOT = f"{BASE_INPUT}/frames/frames"
TEST_CSV = f"{BASE_INPUT}/test_multilabel.csv"

CHECKPOINT_DIR = "/kaggle/working/FER_Project/checkpoints/exp39_multilabel_resnet_se_bilstm2_attention_mtcnn"
CKPT_PATH = f"{CHECKPOINT_DIR}/best_model.pt"

BBOX_CACHE_PATH = "/kaggle/working/FER_Project/processed_exp39/face_bboxes.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_WORKERS = 2


@torch.no_grad()
def main() -> None:
    print("Using device:", DEVICE)
    print("Checkpoint:", CKPT_PATH)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    num_frames = ckpt.get("num_frames", 24)
    image_size = ckpt.get("image_size", 224)
    thresholds = ckpt.get("thresholds", [0.5, 0.5, 0.5, 0.5])
    face_margin = float(ckpt.get("face_margin", 0.20))
    use_face_detection = bool(ckpt.get("use_face_detection", True))
    print("Using thresholds:", thresholds)
    print("use_face_detection:", use_face_detection, "face_margin:", face_margin)

    face_detector = (
        MTCNNFaceDetector(
            device="cuda" if torch.cuda.is_available() else "cpu",
            min_face_size=40,
            margin=face_margin,
            cache_path=BBOX_CACHE_PATH,
        )
        if use_face_detection
        else None
    )

    test_ds = DAiSEEMultilabelDatasetExp39(
        TEST_CSV, FRAMES_ROOT,
        num_frames=num_frames, image_size=image_size,
        train=False, face_detector=face_detector, seed=42,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = ResNet50SEBiLSTM2AttentionMultilabelExp39(
        num_labels=4, pretrained=False,
        hidden_size=256, num_layers=2,
        dropout=0.5, freeze_backbone=False,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_probs, all_labels, all_clip_ids = [], [], []
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

    if face_detector is not None:
        face_detector.save_cache()

    thr = np.array(thresholds).reshape(1, -1)
    preds = (all_probs >= thr).astype(int)

    label_acc = (preds == all_labels).mean(axis=0)
    exact_match = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    samples_f1 = f1_score(all_labels, preds, average="samples", zero_division=0)
    weighted_f1 = f1_score(all_labels, preds, average="weighted", zero_division=0)
    report = classification_report(
        all_labels, preds, target_names=LABEL_COLS, output_dict=True, zero_division=0
    )

    final_report = {
        "model": "Exp39 Multilabel ResNet50 + SE + 2-layer BiLSTM + Attention + 24 Frames + MTCNN",
        "thresholds": {LABEL_COLS[i]: float(thresholds[i]) for i in range(len(LABEL_COLS))},
        "exact_match_accuracy": float(exact_match),
        "per_label_accuracy": {LABEL_COLS[i]: float(label_acc[i]) for i in range(len(LABEL_COLS))},
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "samples_f1": float(samples_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "label_cols": LABEL_COLS,
    }

    print("\n===== EXP39 MULTILABEL TEST RESULTS =====")
    print("Exact Match Accuracy:", exact_match)
    print("Per-label Accuracy:", final_report["per_label_accuracy"])
    print("Macro F1:", macro_f1)
    print("Micro F1:", micro_f1)
    print("Samples F1:", samples_f1)
    print("Weighted F1:", weighted_f1)
    print("Thresholds:", final_report["thresholds"])

    print("\nClassification report:")
    print(classification_report(all_labels, preds, target_names=LABEL_COLS, zero_division=0))

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{CHECKPOINT_DIR}/test_report.json", "w") as fh:
        json.dump(final_report, fh, indent=2)

    with open(f"{CHECKPOINT_DIR}/test_report.txt", "w") as fh:
        fh.write("Exp39 Multilabel ResNet50 + SE + 2-layer BiLSTM + Attention + 24 Frames + MTCNN\n")
        fh.write("=" * 80 + "\n")
        fh.write(f"Exact Match Accuracy: {exact_match:.6f}\n")
        fh.write(f"Macro F1: {macro_f1:.6f}\n")
        fh.write(f"Micro F1: {micro_f1:.6f}\n")
        fh.write(f"Samples F1: {samples_f1:.6f}\n")
        fh.write(f"Weighted F1: {weighted_f1:.6f}\n")
        fh.write(f"Thresholds: {final_report['thresholds']}\n\n")
        fh.write(classification_report(all_labels, preds, target_names=LABEL_COLS, zero_division=0))

    pred_df = pd.DataFrame({"ClipID": all_clip_ids})
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
