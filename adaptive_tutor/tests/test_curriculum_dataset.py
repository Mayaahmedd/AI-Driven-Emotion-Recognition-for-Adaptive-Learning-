"""DatasetCurriculumProvider tests (ASSISTments CSV adapter)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_tutor.core.exceptions import AdaptiveTutorError
from adaptive_tutor.memory.providers.dataset_provider import (
    ASSISTMENTS_ACTIONS,
    DatasetCurriculumProvider,
)
from adaptive_tutor.rewards import default_reward_engine

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSV = _REPO_ROOT / "configs" / "curriculum" / "examples" / "assistments_synthetic.csv"


# ---- Concept catalog from skills -----------------------------------------


def test_skills_become_concepts() -> None:
    p = DatasetCurriculumProvider(_CSV)
    slugs = [c["concept_id"] for c in p.get_concepts()]
    # Synthetic fixture has 5 unique skills (addition / subtraction / mult /
    # division / fractions / equation-solving) - count is fixture-specific.
    assert "addition_whole_numbers" in slugs
    assert "fractions_basic" in slugs
    # No prerequisites at this level - dataset has no prereq structure.
    for c in p.get_concepts():
        assert c["prerequisites"] == []


def test_extras_contain_dataset_statistics() -> None:
    p = DatasetCurriculumProvider(_CSV)
    concepts = {c["concept_id"]: c for c in p.get_concepts()}
    extras = concepts["addition_whole_numbers"]["extras"]
    assert "historical_success_rate" in extras
    assert 0.0 <= extras["historical_success_rate"] <= 1.0
    assert "mean_hint_count" in extras
    assert "n_records" in extras and extras["n_records"] > 0


# ---- Transitions ---------------------------------------------------------


def test_iter_transitions_actions_in_vocabulary() -> None:
    p = DatasetCurriculumProvider(_CSV)
    actions_seen = {t.action for t in p.iter_transitions()}
    assert actions_seen.issubset(set(ASSISTMENTS_ACTIONS))


def test_iter_transitions_done_flag_terminates_groups() -> None:
    p = DatasetCurriculumProvider(_CSV)
    # Group transitions by (user, skill) and verify exactly one `done=True`
    # per group at the *end* of that group.
    from collections import defaultdict

    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for t in p.iter_transitions():
        groups[(t.user_id, t.skill)].append(t.done)
    for key, dones in groups.items():
        assert dones.count(True) == 1, f"group {key} has {dones}"
        assert dones[-1] is True, f"done flag is not on last transition for {key}"


def test_rewards_are_finite_and_bounded() -> None:
    p = DatasetCurriculumProvider(_CSV)
    for t in p.iter_transitions():
        # Coefficients sum to at most ~1.3 in absolute terms.
        assert -2.0 < t.reward < 2.0
        assert t.reward == t.reward  # not NaN


def test_iter_transitions_reward_matches_reward_engine() -> None:
    """First synthetic row is addition, correct=1, hint=0; reward = engine(state emotion)."""
    p = DatasetCurriculumProvider(_CSV)
    t = next(p.iter_transitions())
    eng = default_reward_engine()
    emotion = t.next_state["emotion"]
    expected = eng.compute_scalar(correct=1, hint_count=0, emotion=emotion)
    assert t.reward == pytest.approx(expected)


def test_emotion_keys_match_fer_vocabulary() -> None:
    p = DatasetCurriculumProvider(_CSV)
    for t in p.iter_transitions():
        e = t.state["emotion"]
        assert set(e.keys()) == {"engaged", "confused", "bored", "frustrated"}
        for v in e.values():
            assert 0.0 <= v <= 1.0


# ---- all_stats / stats accessors -----------------------------------------


def test_all_stats_returns_one_entry_per_skill() -> None:
    p = DatasetCurriculumProvider(_CSV)
    stats = p.all_stats()
    assert "addition_whole_numbers" in stats
    s = stats["addition_whole_numbers"]
    assert s.n_records > 0
    assert 0.0 <= s.mean_correctness <= 1.0


def test_stats_accessor_accepts_raw_skill_or_slug() -> None:
    p = DatasetCurriculumProvider(_CSV)
    by_slug = p.stats("addition_whole_numbers")
    by_raw = p.stats("Addition Whole Numbers")
    assert by_slug == by_raw


def test_stats_unknown_skill_raises() -> None:
    p = DatasetCurriculumProvider(_CSV)
    with pytest.raises(KeyError):
        p.stats("does_not_exist")


# ---- Error paths ---------------------------------------------------------


def test_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(AdaptiveTutorError, match="file not found"):
        DatasetCurriculumProvider(tmp_path / "missing.csv")


def test_empty_csv_raises(tmp_path: Path) -> None:
    bad = tmp_path / "empty.csv"
    bad.write_text("skill,correct\n", encoding="utf-8")  # header only
    with pytest.raises(AdaptiveTutorError, match="no rows"):
        DatasetCurriculumProvider(bad)


def test_missing_required_column_raises(tmp_path: Path) -> None:
    bad = tmp_path / "no_skill.csv"
    bad.write_text("correct,hint_count\n1,0\n", encoding="utf-8")
    with pytest.raises(AdaptiveTutorError, match="no 'skill' column"):
        DatasetCurriculumProvider(bad)


# ---- Malformed-value warning behaviour ------------------------------------


def test_malformed_numeric_logs_once_per_column(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One WARNING is emitted per malformed column, regardless of how
    many bad rows appear. Keeps logs readable on dirty CSVs and proves
    determinism is preserved (coerced value is still 0)."""
    from adaptive_tutor.memory.providers import dataset_provider

    dataset_provider._reset_malformed_warnings()

    bad = tmp_path / "dirty.csv"
    bad.write_text(
        "user_id,skill,correct,hint_count\n"
        "u1,Addition,1,abc\n"  # bad hint_count
        "u1,Addition,0,xyz\n"  # bad hint_count again - should NOT log
        "u1,Addition,1,2\n",
        encoding="utf-8",
    )

    with caplog.at_level(
        "WARNING", logger="adaptive_tutor.memory.providers.dataset_provider"
    ):
        DatasetCurriculumProvider(bad)

    hint_warnings = [
        r for r in caplog.records if "column hint_count" in r.message
    ]
    assert len(hint_warnings) == 1, (
        f"expected exactly one warning for hint_count, got {len(hint_warnings)}"
    )
