"""
Compare exp41 (stability-enhanced) against the exp40 variant C baseline.

Reads
-----
exp40-C test report:
    /kaggle/working/FER_Project/checkpoints/exp40_imbalance_ablation/
        variant_C_combined/test_report.json

exp41 test reports (whichever modes have been evaluated):
    /kaggle/working/FER_Project/checkpoints/exp41_stability_enhanced/
        test_report_best_ema.json
        test_report_best_live.json
        test_report_topk_ensemble.json

Also reads ``train_log.json`` from exp41 to extract the per-epoch
EMA-vs-LIVE validation Macro F1 deltas.

Writes
------
    /kaggle/working/FER_Project/checkpoints/exp41_stability_enhanced/
        comparison_exp41_vs_exp40c.json
        comparison_exp41_vs_exp40c.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]

EXP40_C_REPORT = Path(
    "/kaggle/working/FER_Project/checkpoints/exp40_imbalance_ablation/"
    "variant_C_combined/test_report.json"
)
EXP41_DIR = Path("/kaggle/working/FER_Project/checkpoints/exp41_stability_enhanced")

EXP41_REPORTS = {
    "best_ema": EXP41_DIR / "test_report_best_ema.json",
    "best_live": EXP41_DIR / "test_report_best_live.json",
    "topk_ensemble": EXP41_DIR / "test_report_topk_ensemble.json",
}


def _read(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r") as fh:
        return json.load(fh)


def _per_label_f1(report: Optional[dict]) -> dict:
    if report is None:
        return {lab: float("nan") for lab in LABEL_COLS}
    cr = report.get("classification_report", {})
    return {lab: float(cr.get(lab, {}).get("f1-score", float("nan"))) for lab in LABEL_COLS}


def _scalar(report: Optional[dict], key: str) -> float:
    if report is None:
        return float("nan")
    return float(report.get(key, float("nan")))


def main():
    exp40c = _read(EXP40_C_REPORT)
    if exp40c is None:
        print(f"[warn] exp40-C report not found at {EXP40_C_REPORT}")

    exp41 = {mode: _read(p) for mode, p in EXP41_REPORTS.items()}
    available_modes = [m for m, r in exp41.items() if r is not None]
    if not available_modes:
        raise FileNotFoundError(
            "No exp41 test reports found. Run evaluate_exp41.py at least once."
        )

    # ---- per-mode summary --------------------------------------------------
    summary = {
        "exp40c": {
            "macro_f1": _scalar(exp40c, "macro_f1"),
            "micro_f1": _scalar(exp40c, "micro_f1"),
            "weighted_f1": _scalar(exp40c, "weighted_f1"),
            "samples_f1": _scalar(exp40c, "samples_f1"),
            "exact_match_accuracy": _scalar(exp40c, "exact_match_accuracy"),
            "per_label_f1": _per_label_f1(exp40c),
        },
        "exp41": {
            mode: {
                "macro_f1": _scalar(exp41[mode], "macro_f1"),
                "micro_f1": _scalar(exp41[mode], "micro_f1"),
                "weighted_f1": _scalar(exp41[mode], "weighted_f1"),
                "samples_f1": _scalar(exp41[mode], "samples_f1"),
                "exact_match_accuracy": _scalar(exp41[mode], "exact_match_accuracy"),
                "per_label_f1": _per_label_f1(exp41[mode]),
            }
            for mode in available_modes
        },
        "deltas": {
            mode: {
                "macro_f1_delta": _scalar(exp41[mode], "macro_f1") - _scalar(exp40c, "macro_f1"),
                "micro_f1_delta": _scalar(exp41[mode], "micro_f1") - _scalar(exp40c, "micro_f1"),
                "weighted_f1_delta": _scalar(exp41[mode], "weighted_f1") - _scalar(exp40c, "weighted_f1"),
                "samples_f1_delta": _scalar(exp41[mode], "samples_f1") - _scalar(exp40c, "samples_f1"),
                "exact_match_delta": _scalar(exp41[mode], "exact_match_accuracy") - _scalar(exp40c, "exact_match_accuracy"),
                "per_label_f1_delta": {
                    lab: _per_label_f1(exp41[mode])[lab] - _per_label_f1(exp40c)[lab]
                    for lab in LABEL_COLS
                },
            }
            for mode in available_modes
        },
    }

    # ---- per-epoch EMA vs LIVE on validation -------------------------------
    log_path = EXP41_DIR / "train_log.json"
    ema_vs_live = None
    if log_path.exists():
        log = json.loads(log_path.read_text())
        rows = []
        for row in log:
            live = row["live"]["macro_f1"]
            ema = row["ema"]["macro_f1"]
            rows.append({"epoch": int(row["epoch"]), "live": live, "ema": ema, "delta": ema - live})
        if rows:
            deltas = np.array([r["delta"] for r in rows], dtype=np.float32)
            ema_vs_live = {
                "per_epoch": rows,
                "mean_delta_macro_f1_ema_minus_live": float(deltas.mean()),
                "max_delta_macro_f1_ema_minus_live": float(deltas.max()),
                "min_delta_macro_f1_ema_minus_live": float(deltas.min()),
                "last_epoch_delta": float(deltas[-1]),
                "best_live_macro_f1": float(max(r["live"] for r in rows)),
                "best_ema_macro_f1": float(max(r["ema"] for r in rows)),
            }

    summary["ema_vs_live_validation"] = ema_vs_live

    (EXP41_DIR / "comparison_exp41_vs_exp40c.json").write_text(json.dumps(summary, indent=2))

    # ---- markdown report ---------------------------------------------------
    lines = ["# Exp41 — Stability-Enhanced Training vs Exp40-C Baseline\n"]

    lines.append("## Test-set summary\n")
    lines.append(
        "| System | Macro F1 | Micro F1 | Weighted F1 | Samples F1 | Exact Match |"
    )
    lines.append("|---|---|---|---|---|---|")

    def _row(name, d):
        return (
            f"| {name} | {d['macro_f1']:.4f} | {d['micro_f1']:.4f} | "
            f"{d['weighted_f1']:.4f} | {d['samples_f1']:.4f} | "
            f"{d['exact_match_accuracy']:.4f} |"
        )

    lines.append(_row("exp40-C (combined imbalance baseline)", summary["exp40c"]))
    for mode in available_modes:
        lines.append(_row(f"exp41 ({mode})", summary["exp41"][mode]))
    lines.append("")

    lines.append("## Per-label F1 (test set)\n")
    lines.append("| System | " + " | ".join(LABEL_COLS) + " |")
    lines.append("|---|" + "|".join(["---"] * len(LABEL_COLS)) + "|")

    def _plf_row(name, pl):
        return f"| {name} | " + " | ".join(f"{pl[lab]:.4f}" for lab in LABEL_COLS) + " |"

    lines.append(_plf_row("exp40-C", summary["exp40c"]["per_label_f1"]))
    for mode in available_modes:
        lines.append(_plf_row(f"exp41 ({mode})", summary["exp41"][mode]["per_label_f1"]))
    lines.append("")

    lines.append("## Test-set deltas (exp41 mode − exp40-C)\n")
    lines.append("| Mode | ΔMacro F1 | ΔMicro F1 | ΔWeighted F1 | ΔSamples F1 | ΔExact Match |")
    lines.append("|---|---|---|---|---|---|")
    for mode in available_modes:
        d = summary["deltas"][mode]
        lines.append(
            f"| exp41 ({mode}) | {d['macro_f1_delta']:+.4f} | {d['micro_f1_delta']:+.4f} | "
            f"{d['weighted_f1_delta']:+.4f} | {d['samples_f1_delta']:+.4f} | "
            f"{d['exact_match_delta']:+.4f} |"
        )
    lines.append("")

    if ema_vs_live is not None:
        lines.append("## EMA vs non-EMA (validation, per epoch)\n")
        lines.append(f"- best LIVE validation Macro F1 : **{ema_vs_live['best_live_macro_f1']:.4f}**")
        lines.append(f"- best EMA  validation Macro F1 : **{ema_vs_live['best_ema_macro_f1']:.4f}**")
        lines.append(f"- mean per-epoch Δ (EMA − LIVE): {ema_vs_live['mean_delta_macro_f1_ema_minus_live']:+.4f}")
        lines.append(f"- max  per-epoch Δ (EMA − LIVE): {ema_vs_live['max_delta_macro_f1_ema_minus_live']:+.4f}")
        lines.append(f"- last per-epoch Δ (EMA − LIVE): {ema_vs_live['last_epoch_delta']:+.4f}")
        lines.append("")
        lines.append("| Epoch | LIVE Macro F1 | EMA Macro F1 | Δ (EMA − LIVE) |")
        lines.append("|---|---|---|---|")
        for r in ema_vs_live["per_epoch"]:
            lines.append(
                f"| {r['epoch']} | {r['live']:.4f} | {r['ema']:.4f} | {r['delta']:+.4f} |"
            )
        lines.append("")

    out_md = EXP41_DIR / "comparison_exp41_vs_exp40c.md"
    out_md.write_text("\n".join(lines))

    print("Wrote:")
    print(" ", EXP41_DIR / "comparison_exp41_vs_exp40c.json")
    print(" ", out_md)


if __name__ == "__main__":
    main()
