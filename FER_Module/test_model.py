from src.datasets.dataset import DAISEEMultilabelDataset
from src.models.model import ResNet50BiLSTM

ds = DAISEEMultilabelDataset(
    csv_path="data/metadata/train_multilabel.csv",
    frames_root="data/frames",
    image_size=224,
    num_frames=16,
)

sample = ds[0]
frames = sample["frames"].unsqueeze(0)  # [1, 16, 3, 224, 224]

model = ResNet50BiLSTM(num_classes=4)
out = model(frames)

print("Input shape:", frames.shape)
print("Output shape:", out.shape)
print("Output:", out)