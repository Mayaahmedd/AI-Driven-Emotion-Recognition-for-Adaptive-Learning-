"""
DAiSEE multilabel dataset for exp41 (stability-enhanced).

This is a faithful copy of the exp38 dataset. The exp41 experiment isolates
training-stability improvements (EMA, cosine LR with warmup, optional
top-k ensemble); the data pipeline is held fixed at the exp38 baseline so
that any Macro-F1 improvement is attributable to the training-time
interventions, not to the data pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]


class DAiSEEMultilabelDatasetExp41(Dataset):
    def __init__(self, csv_path, frames_root, num_frames=24, image_size=224, train=True):
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.frames_root = Path(frames_root)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.train = bool(train)

        assert "ClipID" in self.df.columns
        for col in LABEL_COLS:
            assert col in self.df.columns

    def __len__(self):
        return len(self.df)

    def _find_clip_dir(self, clip_id):
        clip_id = str(clip_id)
        clip_stem = Path(clip_id).stem
        for c in (self.frames_root / clip_id, self.frames_root / clip_stem):
            if c.exists() and c.is_dir():
                return c
        matches = [m for m in self.frames_root.rglob(clip_stem) if m.is_dir()]
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not find frames folder for ClipID={clip_id}")

    def _augment(self, img):
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-15, 15)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if np.random.rand() < 0.25:
            h, w = img.shape[:2]
            angle = np.random.uniform(-7, 7)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if np.random.rand() < 0.15:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        if np.random.rand() < 0.25:
            h, w = img.shape[:2]
            crop_ratio = np.random.uniform(0.90, 1.0)
            new_h = int(h * crop_ratio)
            new_w = int(w * crop_ratio)
            y1 = np.random.randint(0, h - new_h + 1)
            x1 = np.random.randint(0, w - new_w + 1)
            img = img[y1:y1 + new_h, x1:x1 + new_w]
            img = cv2.resize(img, (self.image_size, self.image_size))
        return img

    def _load_frames(self, clip_dir):
        frame_paths = sorted(
            list(clip_dir.glob("*.jpg"))
            + list(clip_dir.glob("*.png"))
            + list(clip_dir.glob("*.jpeg"))
        )
        if len(frame_paths) == 0:
            raise FileNotFoundError(f"No frames found in {clip_dir}")
        total = len(frame_paths)

        if self.train and total > self.num_frames:
            max_start = max(total - self.num_frames, 0)
            start = np.random.randint(0, max_start + 1)
            end = min(start + self.num_frames, total)
            indices = np.linspace(start, end - 1, self.num_frames).astype(int)
        else:
            indices = np.linspace(0, total - 1, self.num_frames).astype(int)

        frames = []
        for idx in indices:
            img = cv2.imread(str(frame_paths[idx]))
            if img is None:
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.image_size, self.image_size))
            if self.train:
                img = self._augment(img)
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            frames.append(img)

        frames = np.stack(frames)
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float()
        return frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_id = row["ClipID"]
        frames = self._load_frames(self._find_clip_dir(clip_id))
        labels = torch.tensor(
            row[LABEL_COLS].values.astype(np.float32),
            dtype=torch.float32,
        )
        return {"frames": frames, "labels": labels, "clip_id": str(clip_id)}
