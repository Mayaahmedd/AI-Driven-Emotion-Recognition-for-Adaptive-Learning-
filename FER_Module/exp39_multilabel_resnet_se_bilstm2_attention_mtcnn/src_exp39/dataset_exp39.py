"""
DAiSEE multilabel dataset for exp39.

Changes relative to exp38
-------------------------
1. Augmentation is sampled ONCE per clip and applied identically to every
   frame in the 24-frame sequence, preserving temporal coherence for the
   downstream BiLSTM.
2. Face detection (MTCNN, see ``face_detector_exp39.py``) is applied to the
   anchor frame of each clip; the resulting bounding box is cropped from
   every frame BEFORE resizing and normalisation. If MTCNN finds no face,
   the full frame is used (matching exp38 behaviour for that clip).
3. Every source of randomness inside ``__getitem__`` uses a per-worker
   ``numpy.random.Generator`` seeded from the DataLoader worker seed, so
   augmentations are reproducible given a single top-level seed.
4. ``cv2.imread`` failures no longer silently substitute a zero frame; the
   previous successfully-loaded frame is reused and a warning is logged.

Preserved exactly
-----------------
- Label set ``["Boredom", "Engagement", "Confusion", "Frustration"]`` and
  the binary class order.
- T = 24 temporal length and the ``np.linspace(0, total-1, 24).astype(int)``
  sampling rule.
- ImageNet RGB normalisation, 224x224 input size.
- Augmentation parameter ranges (flip p=0.5, brightness/contrast p=0.5,
  rotation p=0.25 in ±7°, blur p=0.15 with 3x3 kernel, crop p=0.25 with
  ratio 0.90-1.00).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_COLS = ["Boredom", "Engagement", "Confusion", "Frustration"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DAiSEEMultilabelDatasetExp39(Dataset):
    """Multilabel DAiSEE dataset with per-clip augmentation and MTCNN crop.

    Parameters
    ----------
    csv_path
        CSV file with columns ``ClipID, Boredom, Engagement, Confusion,
        Frustration`` (binary 0/1; the ordinal->binary mapping is performed
        upstream following the Malekshahi et al. policy described in the
        thesis).
    frames_root
        Root directory containing one subfolder of decoded JPEG/PNG frames
        per clip.
    num_frames
        Length of the temporal axis fed to the model (default 24).
    image_size
        Spatial resolution after resize and normalisation (default 224).
    train
        If True, applies temporal random-window logic when source frames are
        longer than ``num_frames`` and applies augmentation.
    face_detector
        Optional ``MTCNNFaceDetector`` instance. If None, the full frame is
        used (i.e. behaviour collapses to exp38's spatial pipeline minus the
        per-frame augmentation defect).
    seed
        Base seed used to derive the dataset's NumPy ``Generator``. The
        DataLoader worker seed is combined with this so that augmentations
        are reproducible per top-level seed.
    """

    import pandas as pd  # local import to keep module import light

    def __init__(
        self,
        csv_path: str,
        frames_root: str,
        num_frames: int = 24,
        image_size: int = 224,
        train: bool = True,
        face_detector=None,
        seed: int = 42,
    ):
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.frames_root = Path(frames_root)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.face_detector = face_detector
        self.base_seed = int(seed)

        assert "ClipID" in self.df.columns, "CSV must contain a ClipID column"
        for col in LABEL_COLS:
            assert col in self.df.columns, f"CSV must contain a {col} column"

        # Per-worker RNG; populated lazily on first access so that each
        # DataLoader worker gets its own independent stream.
        self._rng: Optional[np.random.Generator] = None

    # --------------------------------------------------------------- helpers

    def __len__(self) -> int:
        return len(self.df)

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_seed = self.base_seed if worker_info is None else (self.base_seed + int(worker_info.seed) % (2**32))
            self._rng = np.random.default_rng(worker_seed)
        return self._rng

    def _find_clip_dir(self, clip_id) -> Path:
        clip_id = str(clip_id)
        clip_stem = Path(clip_id).stem
        for c in (self.frames_root / clip_id, self.frames_root / clip_stem):
            if c.exists() and c.is_dir():
                return c
        matches = [m for m in self.frames_root.rglob(clip_stem) if m.is_dir()]
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Could not find frames folder for ClipID={clip_id}")

    # ------------------------------------------------------ frame loading

    def _frame_paths(self, clip_dir: Path) -> List[Path]:
        paths = sorted(
            list(clip_dir.glob("*.jpg"))
            + list(clip_dir.glob("*.png"))
            + list(clip_dir.glob("*.jpeg"))
        )
        if not paths:
            raise FileNotFoundError(f"No frames found in {clip_dir}")
        return paths

    def _sampled_indices(self, total: int) -> np.ndarray:
        """Same rule as exp38: linspace over the available frames.

        With the standard offline decoder producing 16 frames per clip and
        ``num_frames=24``, the else branch is taken at both train and inference
        time and produces a deterministic sequence with ~37.5% duplicated
        indices. We preserve this rule for direct comparability with exp38.
        """
        if self.train and total > self.num_frames:
            rng = self._get_rng()
            max_start = max(total - self.num_frames, 0)
            start = int(rng.integers(0, max_start + 1))
            end = min(start + self.num_frames, total)
            return np.linspace(start, end - 1, self.num_frames).astype(int)
        return np.linspace(0, total - 1, self.num_frames).astype(int)

    def _load_raw_frames(self, frame_paths: List[Path], indices: np.ndarray) -> List[np.ndarray]:
        """Load the sampled frames as RGB uint8 at native resolution.

        Unlike exp38, ``None`` reads do not silently produce a zero image. We
        first try to substitute the previous successfully-loaded frame; only
        if no frame has yet been loaded do we fall back to a placeholder.
        """
        out: List[np.ndarray] = []
        last_good: Optional[np.ndarray] = None
        for idx in indices:
            p = frame_paths[int(idx)]
            img = cv2.imread(str(p))
            if img is None:
                if last_good is not None:
                    warnings.warn(f"cv2.imread returned None for {p}; reusing previous frame.")
                    out.append(last_good.copy())
                    continue
                warnings.warn(f"cv2.imread returned None for {p} and no previous frame available; zero-filling.")
                # 224x224 placeholder; will be resized again downstream if needed.
                out.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            last_good = img
            out.append(img)
        return out

    # --------------------------------------------------- augmentation params

    def _sample_clip_aug_params(self) -> Dict[str, float]:
        """Sample ONE augmentation tuple per clip.

        The same tuple is then applied identically to every frame in the
        clip, which preserves the temporal structure the BiLSTM is trained
        to exploit.
        """
        rng = self._get_rng()
        params: Dict[str, float] = {
            "flip": float(rng.random() < 0.5),
            "do_bc": float(rng.random() < 0.5),
            "alpha": float(rng.uniform(0.85, 1.15)),
            "beta": float(rng.uniform(-15.0, 15.0)),
            "do_rot": float(rng.random() < 0.25),
            "angle": float(rng.uniform(-7.0, 7.0)),
            "do_blur": float(rng.random() < 0.15),
            "do_crop": float(rng.random() < 0.25),
            "crop_ratio": float(rng.uniform(0.90, 1.0)),
            "crop_y_frac": float(rng.uniform(0.0, 1.0)),
            "crop_x_frac": float(rng.uniform(0.0, 1.0)),
        }
        return params

    def _apply_clip_aug(self, img: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply the per-clip parameter tuple to a single frame.

        All branches read their parameters from ``params`` rather than drawing
        new random numbers, so a sequence of frames receives a consistent
        transformation.
        """
        if params["flip"] > 0.5:
            img = cv2.flip(img, 1)
        if params["do_bc"] > 0.5:
            img = cv2.convertScaleAbs(img, alpha=params["alpha"], beta=params["beta"])
        if params["do_rot"] > 0.5:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), params["angle"], 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if params["do_blur"] > 0.5:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        if params["do_crop"] > 0.5:
            h, w = img.shape[:2]
            new_h = max(8, int(h * params["crop_ratio"]))
            new_w = max(8, int(w * params["crop_ratio"]))
            y_max = max(h - new_h, 0)
            x_max = max(w - new_w, 0)
            y1 = int(round(params["crop_y_frac"] * y_max))
            x1 = int(round(params["crop_x_frac"] * x_max))
            img = img[y1: y1 + new_h, x1: x1 + new_w]
            img = cv2.resize(img, (self.image_size, self.image_size))
        return img

    # -------------------------------------------------------- main accessor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        clip_id = str(row["ClipID"])
        clip_dir = self._find_clip_dir(clip_id)

        # 1. sample frame indices and load RAW RGB frames at native resolution
        paths = self._frame_paths(clip_dir)
        indices = self._sampled_indices(len(paths))
        raw_frames = self._load_raw_frames(paths, indices)

        # 2. clip-level face detection (one bbox reused across all frames)
        bbox = None
        if self.face_detector is not None:
            bbox = self.face_detector.detect_clip_bbox(clip_id, raw_frames)

        # 3. crop (or full-frame fallback), then resize to image_size
        cropped: List[np.ndarray] = []
        for f in raw_frames:
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                f_use = f[y1:y2, x1:x2]
                if f_use.size == 0:
                    f_use = f
            else:
                f_use = f
            f_use = cv2.resize(f_use, (self.image_size, self.image_size))
            cropped.append(f_use)

        # 4. one augmentation tuple for the whole clip
        params = self._sample_clip_aug_params() if self.train else None

        # 5. apply identical augmentation to every frame, then normalise
        tensors: List[np.ndarray] = []
        for f in cropped:
            if params is not None:
                f = self._apply_clip_aug(f, params)
            f = f.astype(np.float32) / 255.0
            f = (f - IMAGENET_MEAN) / IMAGENET_STD
            tensors.append(f)

        frames = np.stack(tensors, axis=0)  # T, H, W, 3
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # T, 3, H, W

        labels = torch.tensor(
            row[LABEL_COLS].values.astype(np.float32),
            dtype=torch.float32,
        )

        return {
            "frames": frames,
            "labels": labels,
            "clip_id": clip_id,
        }
