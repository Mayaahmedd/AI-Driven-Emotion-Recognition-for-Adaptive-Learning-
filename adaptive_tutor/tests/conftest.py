"""Pytest: isolate experiment JSON cache from developer ~/.cache."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_fd, _CACHE = tempfile.mkstemp(prefix="adaptive_tutor_test_", suffix=".json")
os.close(_fd)
os.unlink(_CACHE)  # store expects to create the file
os.environ["ADAPTIVE_TUTOR_EXPERIMENT_CACHE"] = _CACHE
Path(_CACHE).parent.mkdir(parents=True, exist_ok=True)
