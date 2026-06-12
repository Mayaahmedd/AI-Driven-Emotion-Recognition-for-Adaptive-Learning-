"""
Test-set evaluation for exp41.

Modes
-----
``--mode best_ema``   (default)
    Evaluate the single best-Macro-F1 EMA checkpoint
    (``best_model_ema.pt``).

``--mode best_live``
    Evaluate the single best-Macro-F1 LIVE-weights checkpoint
    (``best_model_live.pt``). Useful for an internal EMA vs non-EMA
    comparison on the test set in addition to the per-epoch comparison
    that already lives in ``train_log.json``.

``--mode topk_ensemble``
    Average the state dicts of the top-3 best-EMA checkpoints
    (``best_ema_top1.pt`` .. ``best_ema_top3.pt``) before evaluation.
    Because every checkpoint has the same architecture, weight-averaging
    is mathematically well defined and produces a single state dict
    that can be loaded into a freshly constructed model.

All modes use the per-label thresholds that were tuned on validation
during training; the test set is never used to retune any threshold.
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

sys.path.append("/kaggle/working/FER_Project/src_exp41")

from dataset_exp41 import DAiSEEMultilabelDatasetExp41, LABEL_COLS
from model_exp41 import ResNet50SEBiLSTM2AttentionMultilabelExp41


BASE_INPUT = "/kaggle/input/datasets/mayaahmedd59/daisee-farmes"
FRAMES_ROOT = f"{BASE_INPUT}/frames/frames"
TEST_CSV = f"{BASE_INPUT}/test_multilabel.csv"

CHECKPOINT_DIR = Path("/kaggle/working/FER_Project/checkpoints/exp41_stability_enhanced")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_WORKERS = 2


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


def average_state_dicts(state_dicts):
    """Average a list of state dicts that share an identical key set.

    Floating-point tensors are averaged; non-float tensors are taken
    from the first state dict verbatim (averaging integer buffers like
    ``num_batches_tracked`` is undefined).
    """
    if not state_dicts:
        raise ValueError("average_state_dicts received an empty list.")
    keys = state_dicts[0].keys()
    avg = {}
    for k in keys:
        ref = state_dicts[0][k]
        if torch.is_tensor(ref) and ref.dtype.is_floating_point:
            stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
            avg[k] = stacked.mean(dim=0).to(ref.dtype)
        else:
            avg[k] = state_dicts[0][k]
    return avg


def load_mode(mode: str):
    if mode == "best_ema":
        ckpt = torch.load(CHECKPOINT_DIR / "best_model_ema.pt", map_location=DEVICE)
        return ckpt["ema_state_dict"], ckpt.get("thresholds", [0.5] * 4), ckpt
    if mode == "best_live":
        ckpt = torch.load(CHECKPOINT_DIR / "best_model_live.pt", map_location=DEVICE)
        return ckpt["model_state_dict"], ckpt.get("thresholds", [0.5] * 4), ckpt
    if mode == "topk_ensemble":
        ckpts = []
        for rank in (1, 2, 3):
            p = CHECKPOINT_DIR / f"best_ema_top{rank}.pt"
            if not p.exists():
                continue
            ckpts.append(torch.load(p, map_location=DEVICE))
        if not ckpts:
            raise FileNotFoundError("No best_ema_topN.pt checkpoints found.")
        averaged = average_state_dicts([c["ema_state_dict"] for c in ckpts])
        # Use the thresholds of the rank-1 checkpoint (highest val Macro F1).
        ref = ckpts[0]
        ref["_ensemble_size"] = len(ckpts)
        ref["_ensemble_epochs"] = [int(c["epoch"]) for c in ckpts]
        ref["_ensemble_val_scores"] = [float(c.get("best_macro_f1_ema", c.get("macro_f1_ema", -1.0))) for c in ckpts]
        return averaged, ref.get("thresholds", [0.5] * 4), ref
    raise ValueError(f"Unknown mode: {mode}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        default="best_ema",
        choices=["best_ema", "best_live", "topk_ensemble"],
    )
    args = ap.parse_args()

    print("=" * 80)
    print(f"EXP41 evaluation — mode = {args.mode}")
    print("=" * 80)

    state_dict, thresholds, info = load_mode(args.mode)
    print("Using thresholds (frozen from validation tuning):", thresholds)
    if args.mode == "topk_ensemble":
        print(f"Ensemble size            : {info.get('_ensemble_size')}")
        print(f"Ensemble epochs          : {info.get('_ensemble_epochs')}")
        print(f"Ensemble val Macro F1 (rank ordered): {info.get('_ensemble_val_scores')}")

    test_ds = DAiSEEMultilabelDatasetExp41(
        TEST_CSV, FRAMES_ROOT,
        num_frames=info.get("num_frames", 24),
        image_size=info.get("image_size", 224),
        train=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = ResNet50SEBiLSTM2AttentionMultilabelExp41(
        num_labels=4, pretrained=False,
        hidden_size=256, num_layers=2,
        dropout=0.5, freeze_backbone=False,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_probs, all_labels, all_clip_ids = [], [], []
    for batch in tqdm(test_loader, desc=f"Testing ({args.mode})"):
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
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    samples_f1 = f1_score(all_labels, preds, average="samples", zero_division=0)
    weighted_f1 = f1_score(all_labels, preds, average="weighted", zero_division=0)
    exact_match = accuracy_score(all_labels, preds)
    confusion = per_label_confusion(all_labels, preds)
    cls_report = classification_report(
        all_labels, preds, target_names=LABEL_COLS, output_dict=True, zero_division=0
    )

    final_report = {
        "model": "Exp41 ResNet50 + SE + 2-layer BiLSTM + Attention (stability-enhanced)",
        "mode": args.mode,
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
    if args.mode == "topk_ensemble":
        final_report["ensemble_size"] = info.get("_ensemble_size")
        final_report["ensemble_epochs"] = info.get("_ensemble_epochs")
        final_report["ensemble_val_scores"] = info.get("_ensemble_val_scores")

    print(f"\n----- EXP41 TEST METRICS ({args.mode}) -----")
    print(f"Exact match : {exact_match:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Micro F1    : {micro_f1:.4f}")
    print(f"Samples F1  : {samples_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print("Per-label confusion-style summary:")
    for lab, d in confusion.items():
        print(
            f"  {lab:11s} TP={d['tp']:4d} FP={d['fp']:4d} "
            f"FN={d['fn']:4d} TN={d['tn']:4d} "
            f"P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}"
        )

    out_dir = CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"test_report_{args.mode}.json"
    out_txt = out_dir / f"test_report_{args.mode}.txt"
    out_csv = out_dir / f"test_predictions_{args.mode}.csv"

    out_json.write_text(json.dumps(final_report, indent=2))
    with out_txt.open("w") as fh:
        fh.write(f"Exp41 test report — mode = {args.mode}\n")
        fh.write("=" * 80 + "\n")
        fh.write(f"Exact Match Accuracy: {exact_match:.6f}\n")
        fh.write(f"Macro F1   : {macro_f1:.6f}\n")
        fh.write(f"Micro F1   : {micro_f1:.6f}\n")
        fh.write(f"Samples F1 : {samples_f1:.6f}\n")
        fh.write(f"Weighted F1: {weighted_f1:.6f}\n")
        fh.write(f"Thresholds : {final_report['thresholds']}\n\n")
        fh.write("Per-label confusion-style summary:\n")
        for lab, d in confusion.items():
            fh.write(
                f"  {lab:11s} TP={d['tp']:4d} FP={d['fp']:4d} "
                f"FN={d['fn']:4d} TN={d['tn']:4d} "
                f"P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}\n"
            )
        fh.write("\n")
        fh.write(classification_report(all_labels, preds, target_names=LABEL_COLS, zero_division=0))

    pred_df = pd.DataFrame({"ClipID": all_clip_ids})
    for i, lab in enumerate(LABEL_COLS):
        pred_df[f"true_{lab}"] = all_labels[:, i].astype(int)
        pred_df[f"pred_{lab}"] = preds[:, i].astype(int)
        pred_df[f"prob_{lab}"] = all_probs[:, i]
    pred_df.to_csv(out_csv, index=False)

    print("\nSaved:")
    print(" ", out_json)
    print(" ", out_txt)
    print(" ", out_csv)


if __name__ == "__main__":
    main()
