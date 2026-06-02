# Final Source Cleanup Inventory

**Project:** AI-Driven Emotion Recognition for Adaptive Learning ù RL Module  
**Date:** 2026-05-31  
**Method:** Static + runtime import trace from the seven final thesis driver scripts

**Thesis drivers (roots):**

| Script | Tutors |
|--------|--------|
| `thesis_gain_sensitivity.py` | DQN, ERT, Random |
| `emotion_ablation.py` | DQN, ERT, Random |
| `emotion_id_ablation_study.py` | DQN |
| `strict_emotion_ablation.py` | DQN |
| `dqn_stability_study.py` | DQN |
| `final_dqn_model_study.py` | DQN |
| `ert_action_validation.py` | ERT |

**Legend ù thesis relevance:**

| Tag | Meaning |
|-----|---------|
| `THESIS_DRIVER` | Final thesis study entry point |
| `THESIS_AGENT` | DQN, ERT, or Random implementation |
| `THESIS_CORE` | Environment, reward, config, or metrics required by thesis studies |
| `THESIS_SUPPORT` | Transitive dependency loaded when thesis studies run (not a tutor) |
| `THESIS_ERT_CHECK` | ERT validation helpers called by thesis pipeline |
| `LEGACY` | Superseded script; not part of final thesis evidence |
| `LEGACY_FIGURE` | Old figure generator (pre-cleanup thesis suite) |
| `LEGACY_RUNNER` | General multi-algorithm runner |
| `TEST` | Unit/integration test not imported by thesis studies |
| `ORPHAN` | Editor temp artifact |

**Legend ù recommendation:**

| Recommendation | Meaning |
|----------------|---------|
| **KEEP** | Required for thesis studies to import and run **without code changes** |
| **SAFE_TO_DELETE** | Not required by the final thesis pipeline |

---

## Summary

| Recommendation | Count |
|----------------|------:|
| **KEEP** | 32 |
| **SAFE_TO_DELETE** | 18 |
| **Total `.py` files** | 51 |

After deleting all **SAFE_TO_DELETE** files, the remaining **32 files** are the minimal set needed to reproduce every final thesis experiment (DQN, ERT, Random).

---

## Full inventory

| filename | imported by | thesis relevance | delete/keep recommendation |
|----------|-------------|------------------|---------------------------|
| `__init__.py` | (package root) | Package marker; sets matplotlib cache dir | **KEEP** |
| `thesis_gain_sensitivity.py` | ù | `THESIS_DRIVER` ù main tutor comparison (G0ùG5) | **KEEP** |
| `emotion_ablation.py` | ù | `THESIS_DRIVER` ù observation-level emotion ablation | **KEEP** |
| `emotion_id_ablation_study.py` | ù | `THESIS_DRIVER` ù emotion-ID channel ablation | **KEEP** |
| `strict_emotion_ablation.py` | ù | `THESIS_DRIVER` ù strict MDP factorial ablation | **KEEP** |
| `dqn_stability_study.py` | ù | `THESIS_DRIVER` ù DQN stability / checkpoint protocol | **KEEP** |
| `final_dqn_model_study.py` | `final_ac_ablation.py` | `THESIS_DRIVER` ù frozen 20-seed DQN model | **KEEP** |
| `ert_action_validation.py` | ù | `THESIS_DRIVER` ù ERT action-frequency validation | **KEEP** |
| `agents/dqn_agent.py` | All DQN thesis studies, `evaluation/metrics.py` | `THESIS_AGENT` ù DQN | **KEEP** |
| `agents/rule_based.py` | Gain sensitivity, emotion ablation, ERT validation, `evaluation/metrics.py` | `THESIS_AGENT` ù ERT (`ExpertRuleBasedAgent`) | **KEEP** |
| `agents/random_agent.py` | Gain sensitivity, emotion ablation | `THESIS_AGENT` ù Random | **KEEP** |
| `agents/base_agent.py` | All agent modules | `THESIS_CORE` ù shared agent interface | **KEEP** |
| `agents/__init__.py` | Loaded when any `agents.*` submodule is imported | `THESIS_SUPPORT` ù re-exports DQN, ERT, Random | **KEEP** |
| `config.py` | All thesis studies | `THESIS_CORE` ù seeds, paths, hyperparameters | **KEEP** |
| `mdp_definition.py` | `config.py`, env, agents, reward, metrics | `THESIS_CORE` ù MDP constants, state, action maps | **KEEP** |
| `environment/student_env.py` | All thesis studies | `THESIS_CORE` ù Gymnasium environment | **KEEP** |
| `environment/student_model.py` | `student_env.py`, `population.py`, `fer_adapter.py` | `THESIS_CORE` ù synthetic student / affect dynamics | **KEEP** |
| `environment/population.py` | `student_env.py` | `THESIS_CORE` ù student population generator | **KEEP** |
| `environment/__init__.py` | Loaded when `environment.*` is imported | `THESIS_SUPPORT` ù package re-exports | **KEEP** |
| `reward/reward_function.py` | `student_env.py` | `THESIS_CORE` ù reward computation | **KEEP** |
| `reward/__init__.py` | Loaded when `reward.*` is imported | `THESIS_SUPPORT` ù package re-export | **KEEP** |
| `evaluation/metrics.py` | All thesis studies | `THESIS_CORE` ù learning gain, CI, adaptation metrics | **KEEP** |
| `evaluation/__init__.py` | Loaded when `evaluation.metrics` is imported | `THESIS_SUPPORT` ù imports `metrics` + `plots` | **KEEP** *(import blocker)* |
| `evaluation/plots.py` | `evaluation/__init__.py` | `THESIS_SUPPORT` ù legacy plotting; **not used by thesis study logic** but loaded via package init | **KEEP** *(import blocker)* |
| `explainability/explainer.py` | `student_env.py`, `evaluation/metrics.py` | `THESIS_SUPPORT` ù action reason strings in env info | **KEEP** |
| `explainability/__init__.py` | `student_env.py`, `evaluation/metrics.py` | `THESIS_SUPPORT` ù package re-export | **KEEP** |
| `fer_interface/fer_adapter.py` | `student_env.py` | `THESIS_SUPPORT` ù FER stub (simulation mode when `USE_REAL_FER=False`) | **KEEP** |
| `fer_interface/__init__.py` | `student_env.py` | `THESIS_SUPPORT` ù package re-export | **KEEP** |
| `logging_utils/csv_logger.py` | `evaluation/metrics.py` | `THESIS_SUPPORT` ù episode/step CSV logging during eval | **KEEP** |
| `logging_utils/__init__.py` | `evaluation/metrics.py` | `THESIS_SUPPORT` ù package re-export | **KEEP** |
| `tests/test_expert_rule_tutor.py` | `thesis_gain_sensitivity.py` (lazy import in `validate_ert()`) | `THESIS_ERT_CHECK` ù ERT reachability + rollout checks | **KEEP** |
| `dqn_hyperparameter_study.py` | ù | `LEGACY` ù DQN hyperparameter grid; outputs archived | **SAFE_TO_DELETE** |
| `final_ac_ablation.py` | ù | `LEGACY` ù capstone A vs C ablation; figures removed from thesis bundle | **SAFE_TO_DELETE** |
| `main_experiment.py` | `evaluation/sensitivity_analysis.py` | `LEGACY_RUNNER` ù multi-algo runner (DQN, Rule, Random) | **SAFE_TO_DELETE** |
| `reward_signal_audit.py` | ù | `LEGACY` ù historical PPO vs DQN reward audit | **SAFE_TO_DELETE** |
| `thesis_final_comparison.py` | ù | `LEGACY` ù superseded comparison helper | **SAFE_TO_DELETE** |
| `evaluation/generate_thesis_suite.py` | `evaluation/generate_thesis_final.py` | `LEGACY_FIGURE` ù pre-cleanup multi-algo figure suite (PPO labels) | **SAFE_TO_DELETE** |
| `evaluation/generate_thesis_figures.py` | ù | `LEGACY_FIGURE` ù early thesis figure generator | **SAFE_TO_DELETE** |
| `evaluation/generate_thesis_final.py` | ù | `LEGACY_FIGURE` ù wrapper for `generate_thesis_suite` | **SAFE_TO_DELETE** |
| `evaluation/live_viewer.py` | ù | `LEGACY` ù live training viewer (default algo PPO) | **SAFE_TO_DELETE** |
| `evaluation/sensitivity_analysis.py` | ù | `LEGACY` ù simulator sensitivity sweeps | **SAFE_TO_DELETE** |
| `tests/conftest.py` | pytest only | `TEST` ù pytest fixtures | **SAFE_TO_DELETE** |
| `tests/test_env_step.py` | ù | `TEST` ù env step unit test | **SAFE_TO_DELETE** |
| `tests/test_integration_smoke.py` | ù | `TEST` ù phase-9 smoke test | **SAFE_TO_DELETE** |
| `tests/test_mask_blocks.py` | ù | `TEST` ù action mask unit test | **SAFE_TO_DELETE** |
| `tests/test_reward_signs.py` | ù | `TEST` ù reward sign unit test | **SAFE_TO_DELETE** |
| `tests/test_step1_integration.py` | ù | `TEST` ù step-1 integration test | **SAFE_TO_DELETE** |
| `tests/test_strict_no_emotion.py` | ù | `TEST` ù strict no-emotion unit test | **SAFE_TO_DELETE** |
| `.!25531!dqn_pedagogical_analysis.py` | ù | `ORPHAN` ù editor temp file | **SAFE_TO_DELETE** |
| `.!88766!thesis_gain_sensitivity.py` | ù | `ORPHAN` ù editor temp duplicate | **SAFE_TO_DELETE** |

---

## Minimal KEEP set (32 files)

### Thesis drivers (7)

```
thesis_gain_sensitivity.py
emotion_ablation.py
emotion_id_ablation_study.py
strict_emotion_ablation.py
dqn_stability_study.py
final_dqn_model_study.py
ert_action_validation.py
```

### Agents ù DQN / ERT / Random (5)

```
agents/dqn_agent.py
agents/rule_based.py
agents/random_agent.py
agents/base_agent.py
agents/__init__.py
```

### Core infrastructure (20)

```
__init__.py
config.py
mdp_definition.py
environment/student_env.py
environment/student_model.py
environment/population.py
environment/__init__.py
reward/reward_function.py
reward/__init__.py
evaluation/metrics.py
evaluation/__init__.py
evaluation/plots.py         # loaded via evaluation/__init__.py
explainability/explainer.py
explainability/__init__.py
fer_interface/fer_adapter.py
fer_interface/__init__.py
logging_utils/csv_logger.py
logging_utils/__init__.py
tests/test_expert_rule_tutor.py
```

---

## SAFE_TO_DELETE list (18 files)

```
dqn_hyperparameter_study.py
final_ac_ablation.py
main_experiment.py
reward_signal_audit.py
thesis_final_comparison.py
evaluation/generate_thesis_suite.py
evaluation/generate_thesis_figures.py
evaluation/generate_thesis_final.py
evaluation/live_viewer.py
evaluation/sensitivity_analysis.py
tests/conftest.py
tests/test_env_step.py
tests/test_integration_smoke.py
tests/test_mask_blocks.py
tests/test_reward_signs.py
tests/test_step1_integration.py
tests/test_strict_no_emotion.py
.!25531!dqn_pedagogical_analysis.py
.!88766!thesis_gain_sensitivity.py
```

Deleting these **does not break** any of the seven thesis driver scripts.

---

## Import blockers (optional follow-up)

Bandit+DQN was removed 2026-05-31 (`BANDIT_REMOVAL_VERIFICATION.md`). Remaining non-thesis import side-effects:

| File | Blocker | To delete safely, first edit |
|------|---------|------------------------------|
| `evaluation/plots.py` | `evaluation/__init__.py:2` | Remove `plots` import from `evaluation/__init__.py` |
| `evaluation/__init__.py` | Parent package for `evaluation.metrics` | Keep (or inline metrics import path only) |

---

## Direct thesis usage matrix

| File | Direct thesis driver? | Imported by thesis study? |
|------|:---------------------:|:-------------------------:|
| `thesis_gain_sensitivity.py` | Yes | ù |
| `emotion_ablation.py` | Yes | ù |
| `emotion_id_ablation_study.py` | Yes | ù |
| `strict_emotion_ablation.py` | Yes | ù |
| `dqn_stability_study.py` | Yes | ù |
| `final_dqn_model_study.py` | Yes | ù |
| `ert_action_validation.py` | Yes | ù |
| All other **KEEP** rows | No | Yes (transitive) |
| All **SAFE_TO_DELETE** rows | No | No |

---

## Recommended deletion order

1. **Orphan temps** ù `.!25531!...`, `.!88766!...`
2. **Legacy study scripts** ù `dqn_hyperparameter_study.py`, `final_ac_ablation.py`, `reward_signal_audit.py`, `thesis_final_comparison.py`
3. **Legacy runners / figure generators** ù `main_experiment.py`, `evaluation/generate_thesis_*.py`, `evaluation/live_viewer.py`, `evaluation/sensitivity_analysis.py`
4. **Non-thesis tests** ù all `tests/` except `test_expert_rule_tutor.py`

Re-run thesis import smoke test after deletion:

```bash
export PYTHONPATH="$(pwd)"
python3 -c "
mods = [
  'RL_Module.thesis_gain_sensitivity',
  'RL_Module.emotion_ablation',
  'RL_Module.emotion_id_ablation_study',
  'RL_Module.strict_emotion_ablation',
  'RL_Module.dqn_stability_study',
  'RL_Module.final_dqn_model_study',
  'RL_Module.ert_action_validation',
]
for m in mods: __import__(m)
print('OK')
"
```

---

## Conclusion

- **32 files** form the minimal runnable thesis source tree (DQN + ERT + Random + seven study drivers + shared infrastructure).
- **18 files** are **SAFE_TO_DELETE** without affecting final thesis reproduction.
- **2 optional follow-ups** (`evaluation/plots.py`, `evaluation/__init__.py` plots import) are legacy side-effects only.
