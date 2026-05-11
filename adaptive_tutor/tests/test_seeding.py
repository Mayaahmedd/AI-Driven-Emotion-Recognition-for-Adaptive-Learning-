"""Determinism contract for ``adaptive_tutor.utils.seeding``.

These tests are intentionally cheap and run on every CI invocation. If
any of them flake, *every* other test that relies on RNGs becomes
suspect.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from adaptive_tutor.utils.seeding import SeedConfig, derive_seed, seed_everything


def test_seed_everything_python_random_reproducible() -> None:
    seed_everything(SeedConfig(master_seed=42))
    a = [random.random() for _ in range(5)]
    seed_everything(SeedConfig(master_seed=42))
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_everything_numpy_legacy_reproducible() -> None:
    seed_everything(SeedConfig(master_seed=7))
    a = np.random.rand(3).tolist()
    seed_everything(SeedConfig(master_seed=7))
    b = np.random.rand(3).tolist()
    assert a == b


def test_seed_everything_torch_cpu_reproducible() -> None:
    torch = pytest.importorskip("torch")
    seed_everything(SeedConfig(master_seed=11))
    a = torch.randn(4).tolist()
    seed_everything(SeedConfig(master_seed=11))
    b = torch.randn(4).tolist()
    assert a == b


def test_derive_seed_is_stable_across_calls() -> None:
    # Same (master, namespace) must yield same integer regardless of when
    # we ask. This is the contract used by every per-component RNG.
    assert derive_seed(123, "replay") == derive_seed(123, "replay")


def test_derive_seed_distinguishes_namespaces() -> None:
    assert derive_seed(123, "replay") != derive_seed(123, "simulator")


def test_derive_seed_distinguishes_master() -> None:
    assert derive_seed(1, "x") != derive_seed(2, "x")


def test_derive_seed_in_uint32_range() -> None:
    s = derive_seed(987654321, "anything")
    assert 0 <= s < 2**32


def test_seed_everything_accepts_int_shortcut() -> None:
    cfg = seed_everything(7)
    assert isinstance(cfg, SeedConfig)
    assert cfg.master_seed == 7
