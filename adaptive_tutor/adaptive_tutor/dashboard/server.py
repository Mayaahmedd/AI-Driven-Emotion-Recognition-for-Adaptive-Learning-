"""CLI entry: ``adaptive-tutor-dashboard`` (Phase 13)."""

from __future__ import annotations


def run_uvicorn() -> None:
    import uvicorn

    uvicorn.run(
        "adaptive_tutor.dashboard.api:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


def main() -> None:
    run_uvicorn()


if __name__ == "__main__":
    main()
