"""Offline tools (scaffolds and helpers).

This subpackage holds CLI scripts that are not part of the runtime
training/inference path. They are run by hand to produce artefacts
that the runtime then consumes - for example, a teacher-compatible
curriculum YAML extracted from a public dataset.

Why a separate subpackage:
- keeps the runtime ``adaptive_tutor`` import surface tight,
- documents which scripts are "build-time only",
- makes pytest collection narrower (tests do not import these).
"""
