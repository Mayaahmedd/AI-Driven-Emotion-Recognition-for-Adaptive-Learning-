"""
Oversampled training CSV builder for exp39.

This is a faithful copy of the exp38 dominant-group oversampling rule with
two surface changes:

1. Output path moved to ``processed_exp39/`` so exp38 outputs are not
   overwritten.
2. The ``None`` group multiplier is named explicitly (= 1) so the policy is
   fully documented; exp38 used the same value implicitly.

The oversampling itself (priority order Frustration > Confusion > Boredom >
Engagement; multipliers x1/x2/x3/x4; shuffle seed 42) is unchanged so that
exp39 vs exp38 differences are attributable only to per-clip augmentation +
MTCNN cropping and not to a different training-set composition.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_INPUT = "/kaggle/input/datasets/mayaahmedd59/daisee-farmes"
TRAIN_CSV = f"{BASE_INPUT}/train_multilabel.csv"

OUT_DIR = Path("/kaggle/working/FER_Project/processed_exp39")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "train_multilabel_exp39_oversampled.csv"

LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]


def dominant_group(row) -> str:
    """Priority order: rarest minority first."""
    if row["Frustration"] == 1:
        return "Frustration"
    if row["Confusion"] == 1:
        return "Confusion"
    if row["Boredom"] == 1:
        return "Boredom"
    if row["Engagement"] == 1:
        return "Engagement"
    return "None"


# Multipliers chosen to match exp38 exactly. The "None" group (no positive
# label) is kept once so that genuinely neutral clips remain represented.
MULTIPLIERS = {
    "Engagement": 1,
    "Boredom": 2,
    "Confusion": 3,
    "Frustration": 4,
    "None": 1,
}


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)

    print("Original multilabel positive counts:")
    print(df[LABEL_COLS].sum())

    df["group"] = df.apply(dominant_group, axis=1)

    print("\nOriginal group counts:")
    print(df["group"].value_counts())

    parts = []
    for group, multiplier in MULTIPLIERS.items():
        part = df[df["group"] == group].copy()
        if len(part) == 0:
            continue
        duplicated = pd.concat([part] * multiplier, ignore_index=True)
        parts.append(duplicated)
        print(f"{group}: {len(part)} -> {len(duplicated)}")

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced = balanced.drop(columns=["group"])
    balanced.to_csv(OUT_CSV, index=False)

    print("\nBalanced multilabel positive counts:")
    print(balanced[LABEL_COLS].sum())

    print("\nSaved:")
    print(OUT_CSV)


if __name__ == "__main__":
    main()
