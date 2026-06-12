"""
Build the final A vs B vs C comparison report.

Reads
-----
For every variant V in {A, B, C}:
    checkpoints/exp40_imbalance_ablation/variant_{V}_{name}/best_val_report.json
    checkpoints/exp40_imbalance_ablation/variant_{V}_{name}/test_report.json

Writes
------
    checkpoints/exp40_imbalance_ablation/comparison_summary.json
    checkpoints/exp40_imbalance_ablation/comparison_summary.csv
    checkpoints/exp40_imbalance_ablation/comparison_summary.md

The Markdown file contains the side-by-side metric table plus a
recommendation derived from validation Macro F1. The recommendation rule
is:

  * Pick the variant with the highest validation Macro F1.
  * If two variants are within `TIE_BAND` of each other on validation
    Macro F1, recommend the simpler one (A < B < C in complexity).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RESULTS_ROOT = Path("/kaggle/working/FER_Project/checkpoints/exp40_imbalance_ablation")
LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]

VARIANT_NAMES = {
    "A": "oversampling_only",
    "B": "posweight_only",
    "C": "combined",
}

TIE_BAND = 0.005  # 0.5 F1 ≈ noise band on DAiSEE for 10-seed runs
COMPLEXITY_RANK = {"A": 1, "B": 1, "C": 2}  # combined is more complex than either single switch


def _safe_read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r") as fh:
        return json.load(fh)


def _per_label_f1_from_classification_report(rep):
    return {lab: float(rep[lab]["f1-score"]) for lab in LABEL_COLS}


def _per_label_precision_recall(rep):
    return (
        {lab: float(rep[lab]["precision"]) for lab in LABEL_COLS},
        {lab: float(rep[lab]["recall"]) for lab in LABEL_COLS},
    )


def main():
    rows_val = []
    rows_test = []
    summary = {}

    for variant, name in VARIANT_NAMES.items():
        vdir = RESULTS_ROOT / f"variant_{variant}_{name}"
        val_path = vdir / "best_val_report.json"
        test_path = vdir / "test_report.json"

        val = _safe_read_json(val_path)
        test = _safe_read_json(test_path)

        if val is None:
            print(f"[warn] missing {val_path} — skipping variant {variant} (val).")
        if test is None:
            print(f"[warn] missing {test_path} — skipping variant {variant} (test).")

        summary[variant] = {"name": name, "val": val, "test": test}

        if val is not None:
            per_lab_val = _per_label_f1_from_classification_report(val["classification_report"])
            rows_val.append({
                "variant": variant,
                "name": name,
                "macro_f1": val["macro_f1"],
                "micro_f1": val["micro_f1"],
                "samples_f1": val["samples_f1"],
                "weighted_f1": val["weighted_f1"],
                "exact_match_accuracy": val["exact_match_accuracy"],
                **{f"F1_{lab}": per_lab_val[lab] for lab in LABEL_COLS},
            })

        if test is not None:
            per_lab_test = _per_label_f1_from_classification_report(test["classification_report"])
            p_test, r_test = _per_label_precision_recall(test["classification_report"])
            rows_test.append({
                "variant": variant,
                "name": name,
                "macro_f1": test["macro_f1"],
                "micro_f1": test["micro_f1"],
                "samples_f1": test["samples_f1"],
                "weighted_f1": test["weighted_f1"],
                "exact_match_accuracy": test["exact_match_accuracy"],
                **{f"F1_{lab}": per_lab_test[lab] for lab in LABEL_COLS},
                **{f"P_{lab}": p_test[lab] for lab in LABEL_COLS},
                **{f"R_{lab}": r_test[lab] for lab in LABEL_COLS},
            })

    val_df = pd.DataFrame(rows_val)
    test_df = pd.DataFrame(rows_test)

    # ---- recommendation rule ------------------------------------------------
    recommendation = None
    if not val_df.empty:
        best = val_df.sort_values("macro_f1", ascending=False).iloc[0]
        best_macro = float(best["macro_f1"])
        tied = val_df[val_df["macro_f1"] >= best_macro - TIE_BAND]
        if len(tied) > 1:
            tied = tied.assign(_rank=tied["variant"].map(COMPLEXITY_RANK))
            tied = tied.sort_values(["_rank", "macro_f1"], ascending=[True, False])
            chosen = tied.iloc[0]
            recommendation = {
                "chosen_variant": str(chosen["variant"]),
                "chosen_name": str(chosen["name"]),
                "rule": "tie within TIE_BAND on val Macro F1; chose the simpler imbalance correction.",
                "val_macro_f1_chosen": float(chosen["macro_f1"]),
                "val_macro_f1_best": best_macro,
                "tie_band": TIE_BAND,
            }
        else:
            recommendation = {
                "chosen_variant": str(best["variant"]),
                "chosen_name": str(best["name"]),
                "rule": "highest validation Macro F1 (no tie within TIE_BAND).",
                "val_macro_f1_chosen": best_macro,
                "tie_band": TIE_BAND,
            }

    out_summary = {
        "variants": summary,
        "validation_table": rows_val,
        "test_table": rows_test,
        "recommendation": recommendation,
    }

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "comparison_summary.json").write_text(json.dumps(out_summary, indent=2))
    if not val_df.empty:
        val_df.to_csv(RESULTS_ROOT / "comparison_summary_val.csv", index=False)
    if not test_df.empty:
        test_df.to_csv(RESULTS_ROOT / "comparison_summary_test.csv", index=False)

    # ---- markdown summary ---------------------------------------------------
    lines = ["# Exp40 — Imbalance Correction Ablation\n"]

    lines.append("## Variants\n")
    lines.append("| Variant | Name | use_oversampling | use_pos_weight |")
    lines.append("|---|---|---|---|")
    lines.append("| A | oversampling_only | ✅ | ❌ |")
    lines.append("| B | posweight_only    | ❌ | ✅ |")
    lines.append("| C | combined          | ✅ | ✅ |")
    lines.append("")

    def _md_table(df, kind):
        if df.empty:
            return [f"_No {kind} data available._\n"]
        cols = ["variant", "name", "macro_f1", "micro_f1", "weighted_f1", "samples_f1", "exact_match_accuracy"]
        cols += [f"F1_{lab}" for lab in LABEL_COLS]
        df = df[cols].copy()
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        rows = []
        for _, r in df.iterrows():
            cells = [str(r["variant"]), str(r["name"])]
            for c in cols[2:]:
                cells.append(f"{float(r[c]):.4f}")
            rows.append("| " + " | ".join(cells) + " |")
        return [f"## {kind}\n", header, sep] + rows + [""]

    lines += _md_table(val_df, "Validation")
    lines += _md_table(test_df, "Test")

    if recommendation is not None:
        lines.append("## Recommendation\n")
        lines.append(f"**Chosen variant: {recommendation['chosen_variant']} ({recommendation['chosen_name']})**\n")
        lines.append(f"_Selection rule_: {recommendation['rule']}")
        lines.append(f"_Validation Macro F1 of chosen variant_: {recommendation['val_macro_f1_chosen']:.4f}")
        if "val_macro_f1_best" in recommendation:
            lines.append(f"_Validation Macro F1 of best raw variant_: {recommendation['val_macro_f1_best']:.4f}")
        lines.append(f"_Tie band_: {recommendation['tie_band']:.4f}")

    (RESULTS_ROOT / "comparison_summary.md").write_text("\n".join(lines))

    print("Wrote:")
    print(" ", RESULTS_ROOT / "comparison_summary.json")
    if not val_df.empty:
        print(" ", RESULTS_ROOT / "comparison_summary_val.csv")
    if not test_df.empty:
        print(" ", RESULTS_ROOT / "comparison_summary_test.csv")
    print(" ", RESULTS_ROOT / "comparison_summary.md")

    if recommendation is not None:
        print("\nRecommendation:", recommendation)


if __name__ == "__main__":
    main()
