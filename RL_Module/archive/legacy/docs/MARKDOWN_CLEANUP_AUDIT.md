# Markdown Cleanup Audit

**Scope:** entire `RL_Module/` (excluding `.venv/` third-party packages)  
**Date:** 2026-05-31  
**Status:** audit only — **nothing deleted yet**

---

## Summary

| Metric | Value |
|--------|------:|
| Markdown files found | **17** |
| Total size | **131.9 KB** (135,072 bytes) |
| KEEP_REQUIRED | 11 files · 87.9 KB |
| KEEP_DOCUMENTATION | 2 files · 16.2 KB |
| SAFE_TO_DELETE | 4 files · **44.0 KB** |
| Estimated space saved if SAFE_TO_DELETE removed | **44.0 KB** |

`.venv/` contains additional third-party `.md` files (NumPy, SciPy, PyTorch, etc.) — **out of scope**; do not delete.

---

## Per-file inventory

| Path | Size | Purpose | Referenced by code? | Referenced by thesis? | Generated automatically? | Category |
|------|-----:|---------|:-------------------:|:---------------------:|:------------------------:|----------|
| `figures/thesis_gain_sensitivity/THESIS_GAIN_SENSITIVITY_REPORT.md` | 3.4 KB | Main tutor comparison (DQN / ERT / Random) and gain-ratio sensitivity results | **Yes** — `thesis_gain_sensitivity.py` writes; `thesis_final_comparison.py` reads filename | **Yes** — Ch. 5 main results | **Yes** — study driver | KEEP_REQUIRED |
| `figures/emotion_ablation/EMOTION_ABLATION_REPORT.md` | 6.2 KB | Observation-level emotion ablation methodology and results | **Yes** — `emotion_ablation.py` writes/prints | **Yes** — Ch. 6 ablation | **Yes** — study driver | KEEP_REQUIRED |
| `figures/emotion_id_ablation/EMOTION_ID_ABLATION_REPORT.md` | 5.2 KB | Emotion-ID channel ablation + checkpoint selection results | **Yes** — `emotion_id_ablation_study.py` writes | **Yes** — Ch. 4 model selection | **Yes** — study driver | KEEP_REQUIRED |
| `figures/emotion_update_audit/EMOTION_UPDATE_AUDIT.md` | 8.9 KB | AR(1) affect-dynamics diagnostic audit (methodology validation) | **No** — cited as string in `emotion_id_ablation_study.py` report text only | **Yes** — Ch. 3 methodology | **No** — no generator script in repo; standalone audit artifact | KEEP_REQUIRED |
| `figures/dqn_stability_study/DQN_STABILITY_REPORT.md` | 2.8 KB | DQN training length, checkpoint, and hyperparameter stability results | **Yes** — `dqn_stability_study.py` writes | **Yes** — Ch. 4 experimental design | **Yes** — study driver | KEEP_REQUIRED |
| `figures/final_dqn_model/FINAL_DQN_MODEL_REPORT.md` | 5.6 KB | Final 20-seed frozen DQN model configuration and matched confirmation | **Yes** — `final_dqn_model_study.py` writes | **Yes** — Ch. 4 frozen model | **Yes** — study driver | KEEP_REQUIRED |
| `figures/strict_emotion_ablation/STRICT_EMOTION_ABLATION_REPORT.md` | 2.3 KB | Strict factorial ablation (full / obs-only / strict-off) results | **Yes** — `strict_emotion_ablation.py` writes | **Yes** — Ch. 6 ablation | **Yes** — study driver | KEEP_REQUIRED |
| `figures/strict_emotion_ablation/EMOTION_LEAKAGE_AUDIT.md` | 1.6 KB | Runtime leakage checklist for strict no-emotion MDP | **Yes** — `strict_emotion_ablation.py` writes; `final_ac_ablation.py` references path | **Yes** — appendix / validation | **Yes** — study driver | KEEP_REQUIRED |
| `FINAL_PPO_REMOVAL_REPORT.md` | 5.5 KB | Post-removal verification: PPO scripts deleted, import graph clean | **No** | **No** — repo maintenance record | **Yes** — agent-generated | KEEP_REQUIRED |
| `FINAL_THESIS_FILES.md` | 10.1 KB | Post-cleanup figure inventory (38 files, 8 studies) | **No** | **No** — repo planning | **Yes** — agent-generated | KEEP_REQUIRED |
| `FINAL_THESIS_ARTIFACTS.md` | 20.0 KB | Chapter-mapped thesis artifact inventory with KEEP/DELETE recommendations | **No** | **Likely yes** — dissertation planning / chapter map | **Yes** — agent-generated | KEEP_REQUIRED |
| `README.md` | 3.0 KB | Module setup, install, and run instructions | **No** | **No** | **No** — hand-maintained (contains obsolete PPO commands) | KEEP_DOCUMENTATION |
| `CHANGES.md` | 13.3 KB | Historical audit-fix log across MDP, env, agents, reward | **No** | **No** | **No** — hand-maintained development log | KEEP_DOCUMENTATION |
| `CLEANUP_PREVIEW.md` | 13.1 KB | Pre-deletion size preview (figures/logs/models phases) | **No** | **No** | **Yes** — superseded; cleanup largely complete | SAFE_TO_DELETE |
| `CLEANUP_SOURCE_IMPACT.md` | 12.1 KB | Source-impact preview companion to `CLEANUP_PREVIEW.md` | **No** | **No** | **Yes** — superseded preview | SAFE_TO_DELETE |
| `PPO_REMOVAL_AUDIT.md` | 12.4 KB | Pre-removal PPO dependency audit (transitive import analysis) | **No** | **No** | **Yes** — superseded by `FINAL_PPO_REMOVAL_REPORT.md` | SAFE_TO_DELETE |
| `PPO_REMOVAL_DIFF.md` | 6.5 KB | Interim decoupling diff before final PPO deletion | **No** | **No** | **Yes** — superseded by `FINAL_PPO_REMOVAL_REPORT.md` | SAFE_TO_DELETE |

**Note:** `figures/ert_validation/` has no markdown report (PNG + CSV only). ERT validation is documented via figures and `ert_action_validation.py` JSON output.

---

## 1. KEEP_REQUIRED (11 files · 87.9 KB)

### Thesis study reports (`figures/` — 8 files)

| File | Thesis chapter (inferred) |
|------|----------------------------|
| `figures/thesis_gain_sensitivity/THESIS_GAIN_SENSITIVITY_REPORT.md` | Ch. 5 — main results |
| `figures/emotion_ablation/EMOTION_ABLATION_REPORT.md` | Ch. 6 — observation ablation |
| `figures/emotion_id_ablation/EMOTION_ID_ABLATION_REPORT.md` | Ch. 4 — emotion-ID ablation |
| `figures/emotion_update_audit/EMOTION_UPDATE_AUDIT.md` | Ch. 3 — affect dynamics methodology |
| `figures/dqn_stability_study/DQN_STABILITY_REPORT.md` | Ch. 4 — DQN stability |
| `figures/final_dqn_model/FINAL_DQN_MODEL_REPORT.md` | Ch. 4 — frozen thesis model |
| `figures/strict_emotion_ablation/STRICT_EMOTION_ABLATION_REPORT.md` | Ch. 6 — strict factorial ablation |
| `figures/strict_emotion_ablation/EMOTION_LEAKAGE_AUDIT.md` | Appendix — leakage validation |

### Repo inventory / removal verification (3 files)

| File | Role |
|------|------|
| `FINAL_PPO_REMOVAL_REPORT.md` | Confirms PPO fully removed from import graph |
| `FINAL_THESIS_FILES.md` | Canonical post-cleanup figure file list |
| `FINAL_THESIS_ARTIFACTS.md` | Chapter-mapped artifact inventory for dissertation |

---

## 2. KEEP_DOCUMENTATION (2 files · 16.2 KB)

| File | Role | Action note |
|------|------|-------------|
| `README.md` | Setup, install, experiment entry points | **Update** PPO references when convenient (not part of this cleanup) |
| `CHANGES.md` | Implementation audit trail (bugs fixed, spec alignment) | Useful for reproducibility and code archaeology |

---

## 3. SAFE_TO_DELETE (4 files · 44.0 KB)

| File | Size | Rationale |
|------|-----:|-----------|
| `CLEANUP_PREVIEW.md` | 13.1 KB | Temporary pre-deletion preview; figures cleanup completed |
| `CLEANUP_SOURCE_IMPACT.md` | 12.1 KB | Companion preview; information duplicated in `FINAL_THESIS_FILES.md` / `FINAL_THESIS_ARTIFACTS.md` |
| `PPO_REMOVAL_AUDIT.md` | 12.4 KB | Pre-removal audit; fully superseded by `FINAL_PPO_REMOVAL_REPORT.md` |
| `PPO_REMOVAL_DIFF.md` | 6.5 KB | Interim diff doc; content merged into final PPO removal report |

**No obsolete PPO study reports** remain under `figures/` (PPO figure folders were removed earlier).  
**No archived study markdown** found (`figures/_archive/` absent).

---

## 4. Estimated space saved

| Action | Files | Bytes | Human-readable |
|--------|------:|------:|----------------|
| Delete SAFE_TO_DELETE list | 4 | 45,066 | **44.0 KB** |
| Remaining markdown after deletion | 13 | 90,006 | **87.9 KB** |

Deleting SAFE_TO_DELETE files has **negligible disk impact** (~44 KB) but removes stale planning docs that could confuse future readers.

---

## 5. Recommended final markdown set for thesis submission

Attach these **8 study reports** to the dissertation (main text + appendix). They are the authoritative narratives for each experiment; companion CSVs and PNGs live in the same folders.

```
figures/thesis_gain_sensitivity/THESIS_GAIN_SENSITIVITY_REPORT.md      ? Ch. 5
figures/emotion_update_audit/EMOTION_UPDATE_AUDIT.md                   ? Ch. 3
figures/dqn_stability_study/DQN_STABILITY_REPORT.md                      ? Ch. 4
figures/emotion_id_ablation/EMOTION_ID_ABLATION_REPORT.md              ? Ch. 4
figures/final_dqn_model/FINAL_DQN_MODEL_REPORT.md                        ? Ch. 4
figures/emotion_ablation/EMOTION_ABLATION_REPORT.md                      ? Ch. 6
figures/strict_emotion_ablation/STRICT_EMOTION_ABLATION_REPORT.md        ? Ch. 6
figures/strict_emotion_ablation/EMOTION_LEAKAGE_AUDIT.md                 ? Appendix
```

**Do not submit** (repo-only):

- `FINAL_THESIS_ARTIFACTS.md`, `FINAL_THESIS_FILES.md` — planning inventories
- `FINAL_PPO_REMOVAL_REPORT.md` — engineering cleanup record
- `README.md`, `CHANGES.md` — developer documentation
- All SAFE_TO_DELETE preview/audit files

**ERT validation:** no markdown report exists; cite `figures/ert_validation/ert_action_distribution.png` and `action_frequency_summary.csv` instead.

---

## Code reference map (study reports only)

| Markdown | Generator script |
|----------|------------------|
| `THESIS_GAIN_SENSITIVITY_REPORT.md` | `thesis_gain_sensitivity.py` |
| `EMOTION_ABLATION_REPORT.md` | `emotion_ablation.py` |
| `EMOTION_ID_ABLATION_REPORT.md` | `emotion_id_ablation_study.py` |
| `DQN_STABILITY_REPORT.md` | `dqn_stability_study.py` |
| `FINAL_DQN_MODEL_REPORT.md` | `final_dqn_model_study.py` |
| `STRICT_EMOTION_ABLATION_REPORT.md` | `strict_emotion_ablation.py` |
| `EMOTION_LEAKAGE_AUDIT.md` | `strict_emotion_ablation.py` |
| `EMOTION_UPDATE_AUDIT.md` | *(none in repo — preserve as static artifact)* |

Re-running a study driver **regenerates** its report markdown (except `EMOTION_UPDATE_AUDIT.md`).

---

## Next step (awaiting confirmation)

To apply cleanup, delete the 4 SAFE_TO_DELETE files:

```bash
rm RL_Module/CLEANUP_PREVIEW.md \
   RL_Module/CLEANUP_SOURCE_IMPACT.md \
   RL_Module/PPO_REMOVAL_AUDIT.md \
   RL_Module/PPO_REMOVAL_DIFF.md
```

Optionally update `README.md` to remove obsolete PPO commands (separate task).

**No deletions performed in this audit.**
