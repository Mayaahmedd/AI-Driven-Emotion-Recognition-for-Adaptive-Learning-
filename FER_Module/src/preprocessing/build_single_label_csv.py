from pathlib import Path
import pandas as pd

EMOTION_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]

LABEL_MAP = {
    "Boredom": 0,
    "Engagement": 1,
    "Confusion": 2,
    "Frustration": 3,
}

INPUT_OUTPUT_FILES = [
    ("data/metadata/TrainLabels.csv", "data/metadata/train_single_label.csv"),
    ("data/metadata/ValidationLabels.csv", "data/metadata/val_single_label.csv"),
    ("data/metadata/TestLabels.csv", "data/metadata/test_single_label.csv"),
]


def convert_to_single_label(input_csv: str, output_csv: str) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        print(f"[SKIP] File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    print("Columns found:", df.columns.tolist())

    missing_cols = [c for c in EMOTION_COLS + ["ClipID"] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{input_path} is missing required columns: {missing_cols}"
        )

    df["max_val"] = df[EMOTION_COLS].max(axis=1)
    df["num_max"] = df[EMOTION_COLS].eq(df["max_val"], axis=0).sum(axis=1)

    df_single = df[(df["num_max"] == 1) & (df["max_val"] > 0)].copy()
    df_single["label_name"] = df_single[EMOTION_COLS].idxmax(axis=1)
    df_single["label"] = df_single["label_name"].map(LABEL_MAP)

    df_single = df_single[["ClipID", "label_name", "label"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_single.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Total kept: {len(df_single)}")
    print("Class counts:")
    print(df_single["label_name"].value_counts())


def main() -> None:
    for input_csv, output_csv in INPUT_OUTPUT_FILES:
        convert_to_single_label(input_csv, output_csv)


if __name__ == "__main__":
    main()