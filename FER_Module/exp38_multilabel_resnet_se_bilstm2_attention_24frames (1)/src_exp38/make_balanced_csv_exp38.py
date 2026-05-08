
import pandas as pd
from pathlib import Path

BASE_INPUT = "/kaggle/input/datasets/mayaahmedd59/daisee-farmes"
TRAIN_CSV = f"{BASE_INPUT}/train_multilabel.csv"

OUT_DIR = Path("/kaggle/working/FER_Project/processed_exp38")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "train_multilabel_exp38_oversampled.csv"

LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]


def dominant_group(row):
    if row["Frustration"] == 1:
        return "Frustration"
    if row["Confusion"] == 1:
        return "Confusion"
    if row["Boredom"] == 1:
        return "Boredom"
    if row["Engagement"] == 1:
        return "Engagement"
    return "None"


MULTIPLIERS = {
    "Engagement": 1,
    "Boredom": 2,
    "Confusion": 3,
    "Frustration": 4,
    "None": 1,
}


def main():
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
