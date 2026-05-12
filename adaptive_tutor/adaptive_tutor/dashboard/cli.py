"""Console entry that fixes ``sys.path`` before importing the package.

``pip install -e .`` should register ``adaptive_tutor``, but some environments
still fail on the generated script. This module only uses stdlib first, adds
the project root (folder containing ``pyproject.toml``), then imports the
dashboard server.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    root_s = str(project_root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)

    from adaptive_tutor.dashboard.server import run_uvicorn

    run_uvicorn()


if __name__ == "__main__":
    main()
