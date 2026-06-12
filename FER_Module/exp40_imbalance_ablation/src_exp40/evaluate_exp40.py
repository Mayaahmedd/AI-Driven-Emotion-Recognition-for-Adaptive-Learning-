"""
Test-set evaluation for a single exp40 variant.

Usage
-----
    python evaluate_exp40.py --variant A
    python evaluate_exp40.py --variant B
    python evaluate_exp40.py --variant C

The script loads the best checkpoint for the requested variant from
``checkpoints/exp40_imbalance_ablation/variant_{X}_{name}/best_model.pt``,
applies the per-label thresholds stored inside the checkpoint (these were
tuned on validation, *not* on the test set) and reports all metrics, the
sklearn classification report, and a per-label confusion-style summary
(TP / FP / TN / FN + precision / recall / F1 / accuracy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/kaggle/working/FER_Project/src_exp40")

from dataset_exp40 import DAiSEEMultilabelDatasetExp40, LABEL_COLS
from model_exp40 import ResNet50SEBiLSTM2AttentionMultilabelExp40


BASE_INPUT = "/kaggle/input/datasets/mayaahmedd59/daisee-farmes"
FRAMES_ROOT = f"{BASE_INPUT}/frames/frames"
TEST_CSV = f"{BASE_INPUT}/test_multilabel.csv"

RESULTS_ROOT = Path("/kaggle/working/FER_Project/checkpoints/exp40_imbalance_ablation")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_WORKERS = 2


VARIANT_NAMES = {
    "A": "oversampling_only",
    "B": "posweight_only",
    "C": "combined",
}


def per_label_confusion(labels, preds):
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, required=True, choices=list(VARIANT_NAMES.keys()))
    args = ap.parse_args()

    variant = args.variant
    name = VARIANT_NAMES[variant]
    variant_dir = RESULTS_ROOT / f"variant_{variant}_{name}"
    ckpt_path = variant_dir / "best_model.pt"

    print("=" * 80)
    print(f"EXP40 evaluation — variant {variant} ({name})")
    print("checkpoint:", ckpt_path)
    print("=" * 80)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    num_frames = ckpt.get("num_frames", 24)
    image_size = ckpt.get("image_size", 224)
    thresholds = ckpt.get("thresholds", [0.5, 0.5, 0.5, 0.5])
    print("Using thresholds (from validation tuning, frozen):", thresholds)

    test_ds = DAiSEEMultilabelDatasetExp40(
        TEST_CSV, FRAMES_ROOT,
        num_frames=num_frames, image_size=image_size, train=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = ResNet50SEBiLSTM2AttentionMultilabelExp40(
        num_labels=4, pretrained=False,
        hidden_size=256, num_layers=2,
        dropout=0.5, freeze_backbone=False,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_probs, all_labels, all_clip_ids = [], [], []
    for batch in tqdm(test_loader, desc=f"Testing {variant}"):
        x = batch["frames"].to(DEVICE)
        y = batch["labels"].to(DEVICE)
        probs = torch.sigmoid(model(x))
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_clip_ids.extend(batch["clip_id"])
    all_probs = np.concatenate(all_probs, 0)
    all_labels = np.concatenate(all_labels, 0)

    thr = np.array(thresholds).reshape(1, -1)
    preds = (all_probs >= thr).astype(int)

    label_acc = (preds == all_labels).mean(axis=0)
    exact_match = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    samples_f1 = f1_score(all_labels, preds, average="samples", zero_division=0)
    weighted_f1 = f1_score(all_labels, preds, average="weighted", zero_division=0)
    cls_report = classification_report(
        all_labels, preds, target_names=LABEL_COLS, output_dict=True, zero_division=0
    )
    confusion = per_label_confusion(all_labels, preds)

    final_report = {
        "model": f"Exp40 Variant {variant} ({name})",
        "variant": variant,
        "name": name,
        "use_oversampling": bool(ckpt.get("use_oversampling", False)),
        "use_pos_weight": bool(ckpt.get("use_pos_weight", False)),
        "thresholds": {LABEL_COLS[i]: float(thresholds[i]) for i in range(len(LABEL_COLS))},
        "exact_match_accuracy": float(exact_match),
        "per_label_accuracy": {LABEL_COLS[i]: float(label_acc[i]) for i in range(len(LABEL_COLS))},
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "samples_f1": float(samples_f1),
        "weighted_f1": float(weighted_f1),
        "per_label_confusion": confusion,
        "classification_report": cls_report,
        "label_cols": LABEL_COLS,
    }

    print(f"\n----- VARIANT {variant} TEST METRICS -----")
    print(f"Exact match     : {exact_match:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Micro F1        : {micro_f1:.4f}")
    print(f"Samples F1      : {samples_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")
    print("\nPer-label confusion-style summary:")
    for lab, d in confusion.items():
        print(
            f"  {lab:11s}  TP={d['tp']:4d}  FP={d['fp']:4d}  "
            f"FN={d['fn']:4d}  TN={d['tn']:4d}  "
            f"P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['f1']:.3f}"
        )
    print("\nClassification report:")
    print(classification_report(all_labels, preds, target_names=LABEL_COLS, zero_division=0))

    (variant_dir / "test_report.json").write_text(json.dumps(final_report, indent=2))
    with (variant_dir / "test_report.txt").open("w") as fh:
        fh.write(f"Exp40 Variant {variant} ({name})\n")
        fh.write("=" * 80 + "\n")
        fh.write(f"use_oversampling = {final_report['use_oversampling']}\n")
        fh.write(f"use_pos_weight   = {final_report['use_pos_weight']}\n\n")
        fh.write(f"Exact Match Accuracy: {exact_match:.6f}\n")
        fh.write(f"Macro F1   : {macro_f1:.6f}\n")
        fh.write(f"Micro F1   : {micro_f1:.6f}\n")
        fh.write(f"Samples F1 : {samples_f1:.6f}\n")
        fh.write(f"Weighted F1: {weighted_f1:.6f}\n")
        fh.write(f"Thresholds : {final_report['thresholds']}\n\n")
        fh.write("Per-label confusion-style summary:\n")
        for lab, d in confusion.items():
            fh.write(
                f"  {lab:11s}  TP={d['tp']:4d}  FP={d['fp']:4d}  "
                f"FN={d['fn']:4d}  TN={d['tn']:4d}  "
                f"P={d['precision']:.3f}  R={d['recall']:.3f}  F1={d['f1']:.3f}\n"
            )
        fh.write("\n")
        fh.write(classification_report(all_labels, preds, target_names=LABEL_COLS, zero_division=0))

    pred_df = pd.DataFrame({"ClipID": all_clip_ids})
    for i, lab in enumerate(LABEL_COLS):
        pred_df[f"true_{lab}"] = all_labels[:, i].astype(int)
        pred_df[f"pred_{lab}"] = preds[:, i].astype(int)
        pred_df[f"prob_{lab}"] = all_probs[:, i]
    pred_df.to_csv(variant_dir / "test_predictions.csv", index=False)

    print("\nSaved:")
    print(" ", variant_dir / "test_report.json")
    print(" ", variant_dir / "test_report.txt")
    print(" ", variant_dir / "test_predictions.csv")


if __name__ == "__main__":
    main()
