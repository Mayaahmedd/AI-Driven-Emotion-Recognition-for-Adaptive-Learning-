# Final Thesis Files Inventory

**Scope:** `RL_Module/figures/` after figures-only cleanup  
**Generated:** 2026-05-31  
**Total files:** 38 across 8 studies  
**Thesis tutors:** DQN ù Random ù Expert Rule Tutor (ERT)

No PPO figure outputs. No `final_ac_ablation`. No archived studies. No logs. No per-run or checkpoint dumps.

---

## Study index

| Study folder | Files | Thesis role |
|--------------|------:|-------------|
| `thesis_gain_sensitivity/` | 6 | Main tutor comparison + gain-ratio sensitivity |
| `emotion_ablation/` | 6 | Observation-level emotion ablation |
| `emotion_id_ablation/` | 6 | Emotion-ID channel ablation + model selection |
| `emotion_update_audit/` | 1 | Affect dynamics methodology |
| `ert_validation/` | 2 | ERT baseline validation |
| `dqn_stability_study/` | 5 | DQN training/checkpoint stability |
| `final_dqn_model/` | 6 | Frozen 20-seed thesis DQN model |
| `strict_emotion_ablation/` | 6 | Strict MDP emotion factorial ablation |

**Driver scripts** (outside `figures/`, unchanged):  
`thesis_gain_sensitivity.py` ù `emotion_ablation.py` ù `emotion_id_ablation_study.py` ù `strict_emotion_ablation.py` ù `final_dqn_model_study.py` ù `dqn_stability_study.py` ù `ert_action_validation.py`

---

## 1. Thesis Gain Sensitivity

**Folder:** `figures/thesis_gain_sensitivity/`  
**Compares:** Random vs ERT vs DQN across gain ratios G0ùG5 (G4 = 10:1 default)

| File | Type | Purpose | Why kept |
|------|------|---------|----------|
| `THESIS_GAIN_SENSITIVITY_REPORT.md` | Report | Primary results narrative: G4 ranking, sensitivity table, pairwise tests, conclusions | Main dissertation results section |
| `thesis_gain_overview.png` | Figure | Multi-panel overview: tutor comparison + gain-ratio sensitivity | Primary results figure |
| `ranking_table_g4.csv` | Table | G4 aggregate metrics for Random, ERT, DQN | Source for main comparison table |
| `sensitivity_table.csv` | Table | Learning gain by gain ratio (G0ùG5) | Justifies 10:1 default ratio |
| `comparison_table_all_ratios.csv` | Table | Multi-metric sensitivity across all ratios | Extended sensitivity evidence |
| `pairwise_tests_g4.csv` | Table | Holm-corrected pairwise tests at G4 | Statistical significance at thesis default |

---

## 2. Emotion Ablation (Observation-Only)

**Folder:** `figures/emotion_ablation/`  
**Compares:** Random, ERT, DQN under full-emotion vs knowledge-only observations

| File | Type | Purpose | Why kept |
|------|------|---------|----------|
| `EMOTION_ABLATION_REPORT.md` | Report | Methodology, A vs B results, pairwise tests, thesis conclusions | Ablation chapter narrative |
| `emotion_ablation_overview.png` | Figure | Condition ù tutor metric overview | Primary ablation figure |
| `emotion_ablation_table.png` | Figure | Summary table graphic | Dissertation table figure |
| `emotion_benefit_heatmap.png` | Figure | Per-channel emotion benefit heatmap | Ranks observation channel contributions |
| `emotion_ablation_summary.csv` | Table | Aggregate metrics by condition ù tutor | Source for ablation tables |
| `pairwise_full_vs_knowledge_only.csv` | Table | Holm-corrected A vs B pairwise tests | Statistical evidence for observation ablation |

---

## 3. Emotion-ID Ablation

**Folder:** `figures/emotion_id_ablation/`  
**Isolates:** discrete `emotion_id` observation channel (Condition A vs D)

| File | Type | Purpose | Why kept |
|------|------|---------|----------|
| `EMOTION_ID_ABLATION_REPORT.md` | Report | A vs D results, checkpoint analysis, thesis model decision (drop emotion_id) | Model-selection chapter |
| `emotion_id_ablation_overview.png` | Figure | A vs D comparison overview | Primary figure |
| `volatility_vs_learning_gain.png` | Figure | Seed volatility vs learning gain | Supports variance argument for dropping emotion_id |
| `summary_best_checkpoint.csv` | Table | Best-checkpoint aggregates (A vs D) | Primary aggregate table |
| `pairwise_best_checkpoint.csv` | Table | Holm-corrected A vs D pairwise (best checkpoint) | Statistical comparison |
| `seed_robustness_10_vs_15.csv` | Table | Robustness across 10- vs 15-seed subsets | Validates seed protocol before 20-seed freeze |

---

## 4. Emotion Update Audit

**Folder:** `figures/emotion_update_audit/`  
**Scope:** Read-only audit of AR(1) affect dynamics in `student_model.py`

| File | Type | Purpose | Why kept |
|------|------|---------|----------|
| `EMOTION_UPDATE_AUDIT.md` | Report | Equations, parameters, Monte Carlo stability checks | Methodology appendix; cited by emotion-ID ablation |

---

## 5. ERT Validation

**Folder:** `figures/ert_validation/`  
**Validates:** Expert Rule Tutor action distribution after redesign

| File | Type | Purpose | Why kept |
|------|------|---------|----------|
| `ert_action_distribution.png` | Figure | Mean ERT action frequencies across seeds | Baseline methodology figure |
| `action_frequency_summary.csv` | Table | Aggregate ERT action frequencies | Source for ERT baseline description |

---

## 6. DQN Stability Study

**Folder:** `figures/dqn_stability_study/`  
**Tests:** training length, checkpoint selection, exploration, target update, architecture

| File | Type | Purpose | Why kept |
|------|------|---------|----------|
| `DQN_STABILITY_REPORT.md` | Report | Training-length and hyperparameter screen results, recommendation | Justifies 50k + checkpoint selection protocol |
| `training_length_stability.png` | Figure | Learning gain vs training budget | Primary stability figure |
| `hyperparam_stability_ranking.png` | Figure | Hyperparameter variance ranking | Supports final DQN hyperparameter choice |
| `training_length_summary.csv` | Table | Aggregates by training timesteps | Source for training-length table |
| `hyperparam_summary.csv` | Table | Hyperparameter screen aggregates | Source for hyperparameter table |

---

## 7. Final 20-Seed DQN Model

**Folder:** `figures/final_dqn_model/`  
**Confirms:** frozen `baseline_thesis` model (Tier A rejected) on matched 20 seeds

| File | Type | Purpose | Why kept |
|------|------|---------|----------|
| `FINAL_DQN_MODEL_REPORT.md` | Report | Configuration, 20-seed results, pairwise tests, **final model selection** | Defines the frozen thesis DQN used in all subsequent studies |
| `final_dqn_model_overview.png` | Figure | Tier A vs baseline comparison (n=20) | Model-selection confirmation figure |
| `metrics_table_matched_20.csv` | Table | Aggregate metrics (best checkpoint, n=20) | Primary 20-seed summary table |
| `pairwise_tier_vs_baseline_20seed.csv` | Table | Holm-corrected Tier A vs baseline (n=20) | Statistical confirmation |
| `summary_baseline_best_20.csv` | Table | **Frozen baseline thesis model** summary | Canonical model metrics |
| `baseline_per_run_20.csv` | Table | Per-seed baseline results (n=20) | Reproducibility + per-seed defense evidence |

### Frozen model specification (from report)

- **Observation:** knowledge, engagement, frustration, confusion, boredom (no `emotion_id`)
- **Hyperparameters:** lr=0.001, batch=64, buffer=100k, net=[64,64], expl_frac=0.3, final_eps=0.05, target_update=500
- **Training:** 50,000 steps; best checkpoint from {10k, 20k, 30k, 40k, 50k}
- **Gain ratio:** G4 (10:1)

---

## 8. Strict Emotion Ablation

**Folder:** `figures/strict_emotion_ablation/`  
**Factorial:** A (full) ù B (obs ablation) ù C (strict no-emotion MDP)

| File | Type | Purpose | Why kept |
|------|------|---------|----------|
| `STRICT_EMOTION_ABLATION_REPORT.md` | Report | Three-way decomposition: obs vs dynamics vs combined | Strict ablation narrative |
| `EMOTION_LEAKAGE_AUDIT.md` | Report | Checklist proving Condition C is emotion-leakage-free | Methodology validation for strict no-emotion path |
| `strict_ablation_overview.png` | Figure | A/B/C condition overview | Primary strict ablation figure |
| `emotion_decomposition.png` | Figure | Observation vs dynamics contribution | Decomposition figure |
| `strict_ablation_summary.csv` | Table | Condition aggregates | Source for factorial table |
| `strict_ablation_pairwise.csv` | Table | AùB, BùC, AùC pairwise tests | Statistical decomposition evidence |

---

## Recommended dissertation figure set (12 PNG)

| # | File |
|---|------|
| 1 | `thesis_gain_sensitivity/thesis_gain_overview.png` |
| 2 | `ert_validation/ert_action_distribution.png` |
| 3 | `dqn_stability_study/training_length_stability.png` |
| 4 | `dqn_stability_study/hyperparam_stability_ranking.png` |
| 5 | `emotion_id_ablation/emotion_id_ablation_overview.png` |
| 6 | `emotion_id_ablation/volatility_vs_learning_gain.png` |
| 7 | `final_dqn_model/final_dqn_model_overview.png` |
| 8 | `emotion_ablation/emotion_ablation_overview.png` |
| 9 | `emotion_ablation/emotion_ablation_table.png` |
| 10 | `emotion_ablation/emotion_benefit_heatmap.png` |
| 11 | `strict_emotion_ablation/strict_ablation_overview.png` |
| 12 | `strict_emotion_ablation/emotion_decomposition.png` |

---

## Removed in this cleanup (not present anymore)

| Category | Examples removed |
|----------|------------------|
| Entire studies | `figures/final_ac_ablation/` (14 files) |
| Archived outputs | `figures/_archive/` (was absent) |
| Log files | `thesis_run.log`, `full_run.log`, `baseline_20seed_run.log` |
| Superseded CSVs | `pairwise_tier_vs_baseline_15seed.csv`, `summary_baseline_best_15.csv` |
| Per-run exports | `*_per_run.csv` (except `baseline_per_run_20.csv`) |
| Checkpoint dumps | `checkpoint_eval*.csv`, `checkpoint_selection*.csv` |
| JSON duplicates | `*_report.json`, `emotion_leakage_audit.json` |
| Intermediate analysis | `volatility_per_seed.csv`, `pairwise_tests_all_gains.csv`, etc. |

**Source code, agents, environment, evaluation, models, and repo-level logs were not modified.**

---

## Reproducibility commands

```bash
export PYTHONPATH="$(pwd)"
python -m RL_Module.thesis_gain_sensitivity --phase all
python -m RL_Module.ert_action_validation
python -m RL_Module.dqn_stability_study --phase all
python -m RL_Module.emotion_id_ablation_study --phase all
python -m RL_Module.final_dqn_model_study --phase all
python -m RL_Module.emotion_ablation --phase all
python -m RL_Module.strict_emotion_ablation --phase all
```

Re-running studies regenerates trimmed artifacts (per-run CSVs, JSON, logs). The 38 files above are the **curated thesis evidence layer**.
