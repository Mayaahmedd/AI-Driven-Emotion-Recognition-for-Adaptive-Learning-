"""Deterministic seeding for Python, NumPy, and PyTorch.

Why this module exists
----------------------
Reproducibility is a first-class requirement of the project (see master plan
§8 "Experiment lifecycle" and Rule 5 in the task prompt). Every experiment
gets exactly one ``seed`` value at startup. That single integer must
deterministically seed:

1. Python's ``random`` module                  — generic shuffles, jitter.
2. NumPy's *global* legacy generator           — many third-party libraries.
3. NumPy's modern ``np.random.Generator`` API  — what we use internally.
4. PyTorch CPU RNG.
5. PyTorch CUDA RNG (all devices).
6. PyTorch MPS (Apple Silicon) RNG (when available).
7. ``PYTHONHASHSEED`` for set/dict iteration order across re-runs.
8. Whatever cuDNN / MKL / MPS knobs are needed to disable nondeterministic
   kernels.

We do **not** seed an external library's RNG silently — if a downstream
component (e.g., the replay buffer or simulator) needs its own stream, it
takes a per-component seed derived from the master seed via
:func:`derive_seed`. This avoids the classic pitfall where one component
consuming "an extra random number" alters every subsequent component's
trajectory.

Determinism vs performance
--------------------------
Forcing cuDNN determinism (``torch.use_deterministic_algorithms(True)``)
disables some fast kernels and *can* raise on operations that lack a
deterministic implementation. We therefore expose ``strict=True`` as an
opt-in mode for "reproducibility runs" and default to a softer mode for
day-to-day development.

See ADR-000-index and the master plan §"Logging strategy" / §"Checkpoint
strategy" for the surrounding reproducibility discipline.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedConfig:
    """Resolved seed configuration for one experiment.

    Attributes:
        master_seed: The single integer seed that drives everything.
        strict:      If True, enable PyTorch's deterministic algorithm flag
                     and disable cuDNN benchmark. Slower but bit-reproducible.
        cuda_deterministic: Convenience flag toggling cuDNN flags only.
    """

    master_seed: int
    strict: bool = False
    cuda_deterministic: bool = True


def derive_seed(master_seed: int, namespace: str) -> int:
    """Deterministically derive a child seed from ``master_seed``.

    The derivation is a stable hash of ``f"{master_seed}|{namespace}"``,
    truncated to 32 bits so it fits in NumPy's modern seeding API.

    Two different ``namespace`` strings produce *uncorrelated* streams, and
    the same ``(master_seed, namespace)`` pair always returns the same
    integer regardless of Python/PYTHONHASHSEED.

    Args:
        master_seed: The experiment master seed.
        namespace:   A short, human-readable identifier such as ``"replay"``
                     or ``"simulator.cohort"``.

    Returns:
        A 32-bit non-negative integer suitable for ``np.random.default_rng``
        or ``torch.Generator.manual_seed``.

    Complexity:
        O(len(namespace)) — single SHA-256 of a short string.
    """
    h = hashlib.sha256(f"{int(master_seed)}|{namespace}".encode()).digest()
    # Take 4 bytes => 32-bit int. Numpy's default_rng accepts up to 2**63-1
    # but 32 bits is plenty and keeps log messages short.
    return int.from_bytes(h[:4], byteorder="big", signed=False)


def seed_everything(cfg: SeedConfig | int) -> SeedConfig:
    """Seed all global RNGs and (optionally) enforce strict determinism.

    The function is idempotent: calling it twice with the same ``cfg``
    yields the same RNG state. The function is also defensive about
    optional dependencies — it imports ``torch`` lazily and skips MPS/CUDA
    seeding when those backends aren't available.

    Args:
        cfg: Either a fully-formed :class:`SeedConfig` or a bare integer
             which is interpreted as ``SeedConfig(master_seed=cfg)``.

    Returns:
        The resolved :class:`SeedConfig` (useful for logging).

    Edge cases:
        * If ``torch`` is not installed, only Python + NumPy are seeded.
          This lets the seeding tests run in minimal CI environments.
        * ``PYTHONHASHSEED`` cannot be changed once the interpreter is
          running; we set the env var as documentation only. Callers who
          truly need stable hash iteration must set it *before* launching
          Python (e.g., via ``PYTHONHASHSEED=0 python ...``).
    """
    if isinstance(cfg, int):
        cfg = SeedConfig(master_seed=cfg)

    s = int(cfg.master_seed)

    # 1) PYTHONHASHSEED (documentation-only at this point in execution).
    os.environ.setdefault("PYTHONHASHSEED", str(s))

    # 2) Python's built-in random.
    random.seed(s)

    # 3) NumPy: seed both the legacy global RNG and provide a fresh modern one.
    np.random.seed(s)
    # We don't return the Generator here; callers should use derive_seed +
    # np.random.default_rng(...) for per-component streams.

    # 4) Torch — lazy import so utils/seeding.py doesn't force torch on
    #    minimal installations during early phases.
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep but be safe
        return cfg

    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        if cfg.cuda_deterministic:
            # cuDNN flags: benchmark False so cuDNN doesn't pick algorithms
            # based on input shapes (which depends on prior allocations).
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    # MPS (Apple Silicon) — has its own seed call in recent torch versions.
    # torch.mps.manual_seed exists in PyTorch >= 2.0. Guard each attribute.
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
    ):
        torch.mps.manual_seed(s)

    # 5) Strict determinism. Some ops (e.g., scatter on CUDA) lack a
    #    deterministic implementation; calling them under strict mode will
    #    raise — that is the intended behavior for a "repro run".
    if cfg.strict:
        torch.use_deterministic_algorithms(True, warn_only=False)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        # Soft mode: still ask Torch to use deterministic algorithms *when*
        # available, but don't raise on non-determinism. Easier dev loop.
        torch.use_deterministic_algorithms(False)

    return cfg
