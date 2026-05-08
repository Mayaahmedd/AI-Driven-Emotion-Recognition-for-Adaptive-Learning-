from pathlib import Path
import pandas as pd

EMOTION_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]

FILES = [
    ("data/metadata/TrainLabels.csv", "data/metadata/train_multilabel.csv"),
    ("data/metadata/ValidationLabels.csv", "data/metadata/val_multilabel.csv"),
    ("data/metadata/TestLabels.csv", "data/metadata/test_multilabel.csv"),
]

def convert_file(input_csv: str, output_csv: str) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    df[EMOTION_COLS] = (df[EMOTION_COLS] >= 2).astype(int)

    out_df = df[["ClipID"] + EMOTION_COLS].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(out_df[EMOTION_COLS].sum())

def main():
    for input_csv, output_csv in FILES:
        convert_file(input_csv, output_csv)

if __name__ == "__main__":
    main()