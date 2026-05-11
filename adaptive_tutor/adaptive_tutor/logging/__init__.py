"""adaptive_tutor.logging — unified metric + trace logging sink.

Public surface:

    from adaptive_tutor.logging import ExperimentLogger, LoggerConfig

Everything else is private. This package never imports from anywhere else
in the codebase except ``adaptive_tutor.utils`` — it sits one level above
``utils`` in the dependency DAG.
"""

from adaptive_tutor.logging.logger import ExperimentLogger, LoggerConfig

__all__ = ["ExperimentLogger", "LoggerConfig"]
