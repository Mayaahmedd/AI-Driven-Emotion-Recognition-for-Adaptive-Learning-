"""
MTCNN-based face detector for exp39.

Design rationale
----------------
For each clip we detect a single bounding box (the largest visible face on the
anchor frame, with optional median over a few frames) and apply that *same* box
to every frame in the sampled sequence. Detecting per-frame would re-introduce
temporal inconsistency, defeating the purpose of moving to per-clip
augmentation. The bounding box is also cached to disk so MTCNN is only invoked
once per clip across all epochs.

Fallback
--------
If no face is detected on the anchor frame, the detector returns ``None`` and
the dataset falls back to using the full frame, preserving exp38 behaviour on
clips that MTCNN cannot handle.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


class MTCNNFaceDetector:
    """Lazy-initialised, thread-safe wrapper around facenet-pytorch's MTCNN.

    The detector is lazy because instantiating MTCNN allocates CUDA memory; we
    want this to happen inside each DataLoader worker, not in the main process.
    """

    def __init__(
        self,
        device: str = "cpu",
        min_face_size: int = 40,
        thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7),
        cache_path: Optional[str] = None,
        margin: float = 0.20,
    ):
        self.device = device
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.margin = float(margin)
        self.cache_path = Path(cache_path) if cache_path else None

        self._detector = None
        self._cache: Dict[str, Optional[List[int]]] = {}
        self._cache_dirty = False
        self._lock = threading.Lock()

        if self.cache_path is not None and self.cache_path.exists():
            try:
                with self.cache_path.open("r") as fh:
                    raw = json.load(fh)
                # cache stored as {clip_id: [x1,y1,x2,y2] or null}
                self._cache = {str(k): (list(v) if v is not None else None) for k, v in raw.items()}
                print(f"[MTCNN cache] loaded {len(self._cache)} entries from {self.cache_path}")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[MTCNN cache] WARNING: failed to read cache ({exc}); starting empty.")
                self._cache = {}

    # ------------------------------------------------------------------ utils

    def _ensure_detector(self):
        if self._detector is not None:
            return
        # Lazy import so the dataset module can be imported on systems without
        # facenet-pytorch installed (e.g. linting environments).
        from facenet_pytorch import MTCNN

        self._detector = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=self.min_face_size,
            thresholds=list(self.thresholds),
            post_process=False,
        )

    # --------------------------------------------------------------- caching

    def has(self, clip_id: str) -> bool:
        return str(clip_id) in self._cache

    def get_cached(self, clip_id: str) -> Optional[BBox]:
        v = self._cache.get(str(clip_id), "MISS")
        if v == "MISS":
            return None  # not cached
        if v is None:
            return None  # cached as "no face detected"
        return tuple(v)  # type: ignore[return-value]

    def save_cache(self) -> None:
        if self.cache_path is None or not self._cache_dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w") as fh:
            json.dump(self._cache, fh)
        self._cache_dirty = False
        print(f"[MTCNN cache] wrote {len(self._cache)} entries to {self.cache_path}")

    # ------------------------------------------------------------- detection

    def detect_clip_bbox(
        self,
        clip_id: str,
        frames_rgb: List[np.ndarray],
    ) -> Optional[BBox]:
        """Return one expanded bounding box covering the dominant face in the clip.

        Parameters
        ----------
        clip_id
            Identifier used for caching. Must be unique per clip.
        frames_rgb
            List of RGB frames (H x W x 3, uint8) at native resolution. The
            detector runs on the middle frame only; the resulting box is reused
            for every frame in the clip.
        """
        key = str(clip_id)

        # Cache hit (including cached "no face") returns immediately.
        if key in self._cache:
            v = self._cache[key]
            return tuple(v) if v is not None else None  # type: ignore[return-value]

        # Cache miss → run detector.
        self._ensure_detector()

        # Use the middle frame as the anchor. The middle is preferred over the
        # first frame because participants in DAiSEE sometimes look down or
        # away at the very start of a clip.
        anchor = frames_rgb[len(frames_rgb) // 2]
        H, W = anchor.shape[:2]

        try:
            boxes, probs = self._detector.detect(anchor)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[MTCNN] detection failed for clip {clip_id}: {exc}; falling back to full frame.")
            boxes, probs = None, None

        if boxes is None or len(boxes) == 0:
            with self._lock:
                self._cache[key] = None
                self._cache_dirty = True
            return None

        # Pick the box with the largest area (the dominant face).
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        idx = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes[idx]

        # Symmetric margin expansion around the face.
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        mx = self.margin * bw
        my = self.margin * bh

        ex1 = int(max(0, np.floor(x1 - mx)))
        ey1 = int(max(0, np.floor(y1 - my)))
        ex2 = int(min(W, np.ceil(x2 + mx)))
        ey2 = int(min(H, np.ceil(y2 + my)))

        if ex2 - ex1 < 8 or ey2 - ey1 < 8:
            # Degenerate box → treat as "no face" and fall back.
            with self._lock:
                self._cache[key] = None
                self._cache_dirty = True
            return None

        bbox: BBox = (ex1, ey1, ex2, ey2)

        with self._lock:
            self._cache[key] = list(bbox)
            self._cache_dirty = True

        return bbox
