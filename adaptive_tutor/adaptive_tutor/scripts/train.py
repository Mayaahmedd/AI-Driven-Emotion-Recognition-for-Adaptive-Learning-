"""adaptive-tutor-train — CLI scaffolding (Phase 0 placeholder).

This is a deliberately tiny entry point so that the project is *runnable*
from the moment Phase 0 lands. Later phases will replace the body with the
real orchestration code (see ``adaptive_tutor.orchestration.runner`` once
Phase 14 ships).

For now the script:

1. Parses ``--seed`` and ``--experiment-name``.
2. Seeds everything via :func:`adaptive_tutor.utils.seeding.seed_everything`.
3. Spins up an :class:`ExperimentLogger` and writes a single ``startup`` trace
   record so we can verify end-to-end that the artifact directory is
   created correctly.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from adaptive_tutor import __version__
from adaptive_tutor.logging import ExperimentLogger, LoggerConfig
from adaptive_tutor.utils.seeding import SeedConfig, seed_everything


def _build_experiment_id(name: str, seed: int) -> str:
    """Compose a stable, sortable, machine+human readable experiment id."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}__seed{seed}__{ts}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adaptive-tutor-train",
        description=f"adaptive_tutor v{__version__} — Phase 0 placeholder trainer.",
    )
    parser.add_argument("--experiment-name", default="exp_dev")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B even if WANDB_ENABLED=1.",
    )
    args = parser.parse_args(argv)

    seed_cfg = seed_everything(SeedConfig(master_seed=args.seed, strict=args.strict_determinism))

    log_cfg = LoggerConfig(
        experiment_id=_build_experiment_id(args.experiment_name, seed_cfg.master_seed),
        enable_wandb=False if args.no_wandb else None,  # type: ignore[arg-type]
    )
    # ``enable_wandb`` is computed from env by default; ``--no-wandb`` overrides.
    if args.no_wandb:
        log_cfg = LoggerConfig(experiment_id=log_cfg.experiment_id, enable_wandb=False)

    with ExperimentLogger(log_cfg) as logger:
        logger.log_trace(
            "startup",
            payload={
                "version": __version__,
                "seed": seed_cfg.master_seed,
                "strict": seed_cfg.strict,
                "experiment_name": args.experiment_name,
            },
            step=0,
        )
        logger.log_scalar("env/startup", 1.0, step=0)
        print(
            f"[adaptive-tutor-train] v{__version__} ok | "
            f"experiment_id={log_cfg.experiment_id} | "
            f"run_dir={log_cfg.run_dir}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via ``python -m``
    sys.exit(main())
