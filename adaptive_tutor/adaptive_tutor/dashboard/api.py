"""Read-only FastAPI dashboard (Phase 13)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from adaptive_tutor.core.types import EmotionVector
from adaptive_tutor.dashboard.render import render_dashboard_page, render_explain_form
from adaptive_tutor.experiments.store import get_last_experiment
from adaptive_tutor.explainability import explain_action
from adaptive_tutor.memory.providers.dataset_provider import (
    ASSISTMENTS_ACTIONS,
    DatasetCurriculumProvider,
    SkillStats,
)
from adaptive_tutor.memory.providers.teacher_provider import TeacherCurriculumProvider
from adaptive_tutor.state.state import LearnerState, PerformanceFeatures

_LOG = logging.getLogger("uvicorn.error")


def _skill_stats_from_last_run(concept_index: int) -> SkillStats | None:
    """Cohort :class:`SkillStats` from the latest experiment's resolved CSV paths."""
    exp = get_last_experiment()
    if exp is None:
        return None
    cfg = exp.get("config_resolved") or {}
    dpath = cfg.get("dataset_path")
    tph = cfg.get("teacher_path")
    if not dpath or not tph:
        return None
    try:
        teacher = TeacherCurriculumProvider(Path(str(tph)))
        n = teacher.num_concepts()
        idx = max(0, min(n - 1, int(concept_index)))
        slug = teacher.concept_slug(idx)
        dataset = DatasetCurriculumProvider(Path(str(dpath)))
        return dataset.stats(slug)
    except Exception:
        return None


def _skill_slug_from_last_run(concept_index: int) -> str:
    exp = get_last_experiment()
    if exp is None:
        return "dashboard_demo"
    cfg = exp.get("config_resolved") or {}
    tph = cfg.get("teacher_path")
    if not tph:
        return "dashboard_demo"
    try:
        teacher = TeacherCurriculumProvider(Path(str(tph)))
        idx = max(0, min(teacher.num_concepts() - 1, int(concept_index)))
        return teacher.concept_slug(idx)
    except Exception:
        return "dashboard_demo"


def _learner_state_from_flat(
    *,
    mastery: float,
    concept_index: int,
    concept_id: str,
    engaged: float,
    confused: float,
    bored: float,
    frustrated: float,
    timestep: int,
) -> LearnerState:
    return LearnerState(
        current_concept_id=concept_id,
        current_concept_index=max(0, int(concept_index)),
        mastery=max(0.0, min(1.0, float(mastery))),
        perf=PerformanceFeatures(
            recent_accuracy=max(0.0, min(1.0, mastery)),
            hint_usage=0.1,
            attempts=1,
        ),
        rolling_emotions=EmotionVector(
            engaged=max(0.0, min(1.0, engaged)),
            confused=max(0.0, min(1.0, confused)),
            bored=max(0.0, min(1.0, bored)),
            frustrated=max(0.0, min(1.0, frustrated)),
        ),
        engagement_trend=0.0,
        confusion_trend=0.0,
        frustration_trend=0.0,
        boredom_trend=0.0,
        timestep=max(0, int(timestep)),
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        import adaptive_tutor.dashboard.api as api_mod

        paths = sorted(
            getattr(r, "path", "")
            for r in app.routes
            if getattr(r, "path", None)
        )
        _LOG.info(
            "adaptive_tutor dashboard API loaded from %s (paths: %s)",
            api_mod.__file__,
            ", ".join(paths),
        )
        yield

    app = FastAPI(
        title="Adaptive Tutor (read-only research dashboard)",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_home() -> HTMLResponse:
        return HTMLResponse(render_dashboard_page(get_last_experiment()))

    @app.get("/explain/view", response_class=HTMLResponse)
    @app.get("/explain", response_class=HTMLResponse)
    def explain_view_ui(
        action: str = Query("give_hint"),
        mastery: float = Query(0.35, ge=0.0, le=1.0),
        engaged: float = Query(0.5, ge=0.0, le=1.0),
        confused: float = Query(0.25, ge=0.0, le=1.0),
        frustrated: float = Query(0.15, ge=0.0, le=1.0),
        bored: float = Query(0.1, ge=0.0, le=1.0),
        concept_index: int = Query(0, ge=0),
        timestep: int = Query(0, ge=0),
    ) -> HTMLResponse:
        cid = _skill_slug_from_last_run(concept_index)
        skill_stats = _skill_stats_from_last_run(concept_index)
        st = _learner_state_from_flat(
            mastery=mastery,
            concept_index=concept_index,
            concept_id=cid,
            engaged=engaged,
            confused=confused,
            bored=bored,
            frustrated=frustrated,
            timestep=timestep,
        )
        neutral_p7 = {
            "mask_passed": True,
            "cooldown_blocked": False,
            "prerequisites_met": True,
            "whipsaw_blocked": False,
        }
        expl: dict[str, Any] | None = None
        err: str | None = None
        a = (action or "").strip()
        if a not in ASSISTMENTS_ACTIONS:
            err = (
                f"Unknown action {a!r}. Use one of: {', '.join(ASSISTMENTS_ACTIONS)}."
            )
        else:
            try:
                expl = explain_action(
                    st,
                    a,
                    phase7=neutral_p7,
                    r_components=None,
                    correct=None,
                    skill_stats=skill_stats,
                )
            except Exception as e:
                _LOG.exception("explain_action failed")
                err = str(e)
        return HTMLResponse(
            render_explain_form(
                action=action,
                mastery=mastery,
                engaged=engaged,
                confused=confused,
                bored=bored,
                frustrated=frustrated,
                explanation=expl,
                error=err,
            )
        )

    @app.get("/experiment/latest")
    def experiment_latest() -> dict[str, Any]:
        r = get_last_experiment()
        if r is None:
            raise HTTPException(status_code=404, detail="no experiment stored yet")
        return {"ok": True, "result": r}

    @app.get("/metrics/simulator")
    def metrics_simulator() -> dict[str, Any]:
        r = get_last_experiment()
        if r is None:
            raise HTTPException(status_code=404, detail="no experiment stored yet")
        cm = r.get("comparison_ready_metrics") or {}
        return {
            "learning_rate": r.get("avg_learning_rate", cm.get("learning_rate")),
            "mastery_gain": cm.get("mastery_gain"),
            "frustration_rate": r.get("frustration_rate", cm.get("frustration_rate")),
            "avg_reward": r.get("avg_reward"),
            "hint_efficiency": cm.get("hint_efficiency"),
            "success_rate": cm.get("success_rate"),
        }

    @app.get("/metrics/dataset")
    def metrics_dataset() -> dict[str, Any]:
        r = get_last_experiment()
        if r is None:
            raise HTTPException(status_code=404, detail="no experiment stored yet")
        dm = r.get("dataset_metrics")
        if dm is None:
            raise HTTPException(status_code=404, detail="no dataset metrics on last run")
        return dict(dm)

    @app.get("/explain/action")
    def explain_action_route(
        action: str = Query(..., description="ASSISTments opcode after Phase 7"),
        mastery: float = Query(0.35, ge=0.0, le=1.0),
        concept_index: int = Query(0, ge=0),
        engaged: float = Query(0.5, ge=0.0, le=1.0),
        confused: float = Query(0.25, ge=0.0, le=1.0),
        bored: float = Query(0.1, ge=0.0, le=1.0),
        frustrated: float = Query(0.15, ge=0.0, le=1.0),
        timestep: int = Query(0, ge=0),
        state_json: str | None = Query(
            None,
            description="Optional JSON object overriding scalar query fields",
        ),
    ) -> dict[str, Any]:
        if state_json:
            try:
                data = json.loads(state_json)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"invalid state_json: {e}") from e
            st = _learner_state_from_flat(
                mastery=float(data.get("mastery", mastery)),
                concept_index=int(data.get("current_concept_index", concept_index)),
                concept_id=str(data.get("current_concept_id", _skill_slug_from_last_run(concept_index))),
                engaged=float(data.get("engaged", data.get("rolling_engaged", engaged))),
                confused=float(data.get("confused", confused)),
                bored=float(data.get("bored", bored)),
                frustrated=float(data.get("frustrated", frustrated)),
                timestep=int(data.get("timestep", timestep)),
            )
        else:
            cid = _skill_slug_from_last_run(concept_index)
            st = _learner_state_from_flat(
                mastery=mastery,
                concept_index=concept_index,
                concept_id=cid,
                engaged=engaged,
                confused=confused,
                bored=bored,
                frustrated=frustrated,
                timestep=timestep,
            )
        neutral_p7 = {
            "mask_passed": True,
            "cooldown_blocked": False,
            "prerequisites_met": True,
            "whipsaw_blocked": False,
        }
        a = (action or "").strip()
        if a not in ASSISTMENTS_ACTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown action {a!r}; expected one of {list(ASSISTMENTS_ACTIONS)}",
            )
        try:
            return explain_action(
                st,
                a,
                phase7=neutral_p7,
                r_components=None,
                correct=None,
                skill_stats=_skill_stats_from_last_run(concept_index),
            )
        except Exception as e:
            _LOG.exception("explain_action failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


app = create_app()

__all__ = ["app", "create_app"]
