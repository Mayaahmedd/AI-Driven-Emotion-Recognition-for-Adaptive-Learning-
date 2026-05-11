"""FER client — bridge between the existing FER_Module model and our types.

Design summary
--------------
The FER_Module model (``FER_Module/src/models/model.py::ResNet50BiLSTM``)
expects a tensor of shape ``[B, T, 3, H, W]`` and returns logits of
shape ``[B, 4]`` in the order ``(engaged, confused, bored, frustrated)``
(the order is checked against the trained head in ``EmotionVector``).

For the adaptation engine we want a simple call

    e_t: EmotionVector = client.predict_frames(frames)

where ``frames`` is a NumPy array of shape ``[T, 3, H, W]`` (a single
clip), and we want:

* lazy model loading so that unit tests don't require the ~100 MB
  ResNet50 weights,
* a deterministic ``MockFERClient`` that returns a configurable
  ``EmotionVector`` for tests and offline experiments,
* sigmoid post-processing (the model is *multilabel*; logits become
  independent probabilities, not a softmax),
* CPU/MPS/CUDA-aware device handling (decision 20.13).

We do **not** expose batched prediction at this level — the bandit and
RL loops process one learner clip at a time. A batch API can be added in
later phases when offline-replay evaluation needs it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from adaptive_tutor.core.exceptions import ContractError
from adaptive_tutor.core.types import EmotionVector

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public structural contract for FER clients.
# ---------------------------------------------------------------------------


@runtime_checkable
class FERClient(Protocol):
    """Anything that can turn a clip of frames into an :class:`EmotionVector`.

    A clip is a NumPy array of shape ``[T, 3, H, W]``, dtype ``float32``,
    pixel values in ``[0, 1]`` (already normalized + preprocessed; the
    client is allowed to assume this and is not responsible for cropping,
    face detection, or color conversion). See ``FER_Module`` for the
    preprocessing pipeline that produced these frames.

    The output is the post-sigmoid probability vector. Multilabel: each
    component is independent, not a softmax.
    """

    def predict_frames(self, frames: np.ndarray) -> EmotionVector: ...


# ---------------------------------------------------------------------------
# Production client — lazy-loads FER_Module's ResNet50BiLSTM.
# ---------------------------------------------------------------------------


class FERTorchClient:
    """Production :class:`FERClient` backed by a PyTorch checkpoint.

    The class deliberately does NOT import torch at module import time —
    only inside ``__init__`` — so that early-phase tests and minimal
    environments can ``from adaptive_tutor.fer import FERClient`` without
    a heavy install.

    Parameters
    ----------
    checkpoint_path:
        Path to a state-dict file produced by FER_Module's training
        pipeline. The file is loaded with ``torch.load(..., map_location=
        device)``.
    device:
        ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``. Auto-detection
        prefers CUDA > MPS > CPU.
    num_classes:
        Defensive: must equal 4 to match our :class:`EmotionVector`. If
        the checkpoint disagrees, we raise :class:`ContractError`.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        num_classes: int = 4,
    ) -> None:
        if num_classes != 4:
            raise ContractError(
                f"FERTorchClient supports only the 4-emotion multilabel head "
                f"(got num_classes={num_classes})"
            )
        self._ckpt = Path(checkpoint_path)
        self._device = device
        self._model: object | None = None  # lazy

    # ---- lazy loader ----------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from FER_Module.src.models.model import ResNet50BiLSTM
        except ImportError as e:  # pragma: no cover - depends on user env
            raise ContractError(
                "FERTorchClient requires both torch and the FER_Module "
                "package on PYTHONPATH; install / vendor them before use."
            ) from e

        dev = self._resolve_device(torch)
        model = ResNet50BiLSTM(num_classes=4, pretrained=False)
        state = torch.load(self._ckpt, map_location=dev)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval().to(dev)
        self._model = model
        self._device_str = str(dev)
        _LOG.info("FERTorchClient loaded checkpoint=%s device=%s", self._ckpt, dev)

    @staticmethod
    def _resolve_device(torch_mod: object) -> object:
        torch = torch_mod
        if hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
            return torch.device("cuda")  # type: ignore[attr-defined]
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")  # type: ignore[attr-defined]
        return torch.device("cpu")  # type: ignore[attr-defined]

    # ---- public API -----------------------------------------------------
    def predict_frames(self, frames: np.ndarray) -> EmotionVector:
        """Predict the :class:`EmotionVector` for one clip.

        Args:
            frames: ``[T, 3, H, W]`` float32 array in ``[0, 1]``.

        Returns:
            The sigmoid-activated 4-vector.

        Notes:
            * Forward pass is wrapped in ``torch.inference_mode()`` for
              the latency win on CPU/MPS.
            * Shape checks happen here, not at module import.
        """
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ContractError(
                f"FERTorchClient.predict_frames: expected [T, 3, H, W], "
                f"got shape {frames.shape}"
            )
        self._load()
        import torch  # safe: only reached after _load()

        x = torch.from_numpy(np.ascontiguousarray(frames)).float()
        x = x.unsqueeze(0)  # [1, T, 3, H, W]
        x = x.to(self._device_str)
        with torch.inference_mode():
            logits = self._model(x)  # [1, 4]  # type: ignore[operator]
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
        e = EmotionVector(
            engaged=float(probs[0]),
            confused=float(probs[1]),
            bored=float(probs[2]),
            frustrated=float(probs[3]),
        )
        return e


# ---------------------------------------------------------------------------
# Mock client — for tests and simulator-only training.
# ---------------------------------------------------------------------------


class MockFERClient:
    """Deterministic stand-in for unit tests and simulator-only runs.

    The mock cycles through a user-supplied sequence of
    :class:`EmotionVector` values, repeating the last one once the
    sequence is exhausted. This is enough for:

    * unit tests of the rolling-window cache,
    * simulator runs that don't actually have video frames (the
      simulator drives the emotion vector directly),
    * deterministic CI smoke tests.

    Parameters
    ----------
    sequence:
        Iterable of :class:`EmotionVector` values to return in order. If
        empty, the mock always returns :py:meth:`EmotionVector.neutral`.

    Example::

        client = MockFERClient([
            EmotionVector(0.8, 0.1, 0.0, 0.0),
            EmotionVector(0.5, 0.4, 0.0, 0.1),
        ])
        client.predict_frames(np.zeros((1, 3, 4, 4), np.float32))
        # -> EmotionVector(0.8, 0.1, 0.0, 0.0)
    """

    def __init__(self, sequence: Sequence[EmotionVector] = ()) -> None:
        self._seq: tuple[EmotionVector, ...] = tuple(sequence)
        self._i = 0

    def predict_frames(self, frames: np.ndarray) -> EmotionVector:
        del frames  # mock ignores actual frame content
        if not self._seq:
            return EmotionVector.neutral()
        e = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return e

    def reset(self) -> None:
        """Rewind the cursor; same sequence will replay from the start."""
        self._i = 0
