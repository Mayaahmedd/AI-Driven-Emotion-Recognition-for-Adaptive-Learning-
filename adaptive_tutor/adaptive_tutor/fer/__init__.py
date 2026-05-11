"""adaptive_tutor.fer — adapter to the existing FER_Module multilabel model.

Public surface:

    from adaptive_tutor.fer import (
        FERClient, MockFERClient, FERRollingCache,
    )

The adapter intentionally treats the FER model as a *black box*. Anything
that exposes ``predict_frames(np.ndarray) -> EmotionVector`` is a valid
``FERClient``. This decouples the adaptation engine from any specific
backbone and lets us A/B different FER models without touching the RL
stack.
"""

from adaptive_tutor.fer.cache import FERRollingCache
from adaptive_tutor.fer.client import FERClient, MockFERClient

__all__ = ["FERClient", "FERRollingCache", "MockFERClient"]
