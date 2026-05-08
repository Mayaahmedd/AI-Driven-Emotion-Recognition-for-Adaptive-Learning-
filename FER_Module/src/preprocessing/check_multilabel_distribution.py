import pandas as pd

files = {
    "train": "data/metadata/TrainLabels.csv",
    "val": "data/metadata/ValidationLabels.csv",
    "test": "data/metadata/TestLabels.csv",
}

emotion_cols = ["Boredom", "Engagement", "Confusion", "Frustration"]

for split, path in files.items():
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    bin_df = (df[emotion_cols] >= 2).astype(int)

    print(f"\n--- {split.upper()} ---")
    print("Positive counts per emotion:")
    print(bin_df.sum())

    print("\nNumber of active labels per clip:")
    print(bin_df.sum(axis=1).value_counts().sort_index())