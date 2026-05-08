from src.datasets.dataset import DAISEEMultilabelDataset

ds = DAISEEMultilabelDataset(
    csv_path="data/metadata/train_multilabel.csv",
    frames_root="data/frames",
    image_size=224,
    num_frames=16,
)

sample = ds[0]
print(sample["clip_id"])
print(sample["frames"].shape)
print(sample["labels"])