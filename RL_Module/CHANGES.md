# RL_Module Audit Fixes

Full audit and correction per Phase 0-10 specification.

## mdp_definition.py
- **Bug:** `normalized_knowledge_gain` used `eps=1e-6` instead of spec `1e-8`.
- **Fix:** Changed default epsilon to `1e-8`.
- **Bug:** Emotion normalized to [0,1] in `as_vec()`.
- **Fix:** Store raw `emotion_id` (0-3) in observation vector.

## config.py
- **Bug:** Missing `BEST_ACTION_MAP`, `ACTION_NAMES`, `EMOTION_NAMES`, `TOTAL_TIMESTEPS`, `N_STUDENTS`, `set_all_seeds()`.
- **Fix:** Added all required constants, `PERSISTENT_FLAG_DROPOUT_STEPS=5`, `set_all_seeds()` with CUDA support.

## environment/student_model.py
- **Bug:** Action 7 (strategy) gave zero knowledge gain.
- **Fix:** Added `0.01 * gamma_s` for action 7.
- **Bug:** Engagement only boosted actions 3,1,5; spec requires 3,6,8 at +0.06.
- **Fix:** Implemented full engagement update table.
- **Bug:** Boredom used wrong threshold (0.4), missing easier-when-advanced branch, no default decay.
- **Fix:** Threshold 0.3, branches for motivation/autonomy, easier+knowledge>0.6, else -0.01.

## environment/student_env.py
- **Bug:** `Box(high=1.0)` for all dims; emotion should be 0-3.
- **Fix:** `OBS_HIGH = [1,1,1,1,1,3]`.
- **Bug:** `action_masks()` returned bool; spec requires `get_action_mask()` int8.
- **Fix:** Added `get_action_mask()` returning `np.int8`; bool alias kept for SB3.
- **Bug:** Terminated immediately on persistent flag.
- **Fix:** Dropout only after 5 consecutive flag steps (`PERSISTENT_FLAG_DROPOUT_STEPS`).
- **Bug:** Emotion from FER personality rules, not state thresholds.
- **Fix:** Inline `emotion_from_state()` after transitions.
- **Bug:** DQN crashes on masked actions during training.
- **Fix:** Added `DQNSafeEnv` wrapper to resample blocked actions.
- **Bug:** Render 560px height, QUIT ignored, no headless guard.
- **Fix:** 580px, `_should_quit` on QUIT, try/except on pygame init.
- **Bug:** `info` missing `terminated`, `truncated`, `dropout`.
- **Fix:** Added all fields to step info dict.

## reward/reward_function.py
- **Bug:** Double persistent penalty (in `_frustration_penalty` and again explicitly).
- **Fix:** Removed duplicate `if persistent_flag: r -= W_PERSISTENT`.
- **Bug:** Low frustration still penalized via `W_FRUSTRATION * frustration`.
- **Fix:** Productive frustration elif chain: penalty only if `>0.6` or persistent flag.

## fer_interface/fer_adapter.py
- **Bug:** Simulation used personality-based thresholds.
- **Fix:** State-only: frustration>0.6, confusion>0.5, boredom>0.5.
- **Bug:** Missing `get_emotion(state, fer_raw_output)` public API.
- **Fix:** Added `get_emotion()` and shared `emotion_from_state()`.

## agents/ppo_agent.py
- **Bug:** Unused `BaseCallback` import; separate pi/vf arch only.
- **Fix:** Removed import; `policy_kwargs=dict(net_arch=[256,256,256])`.

## agents/dqn_agent.py
- **Bug:** Plain env raised on masked actions during DQN exploration.
- **Fix:** `DQNSafeEnv` in `make_env()`; `torch.no_grad()`; mask via `action_mask == 0`.

## agents/ppo_dqn_hybrid.py
- **Bug:** Each model trained for only half timesteps.
- **Fix:** Full `total_timesteps` per model.
- **Bug:** `~action_mask` broken for int8 masks.
- **Fix:** `action_mask == 0` zeroing.

## agents/bandit_dqn.py
- **Bug:** Bandit trigger used float delta > 0.3 instead of integer emotion change.
- **Fix:** `abs(eid - prev) >= 1`; context dim 3 per spec.

## agents/rule_based.py
- **Bug:** Mask fallback jumped to first allowed action, not next rule.
- **Fix:** Priority waterfall with per-rule mask check.

## agents/random_agent.py
- **Bug:** Used `np.where(mask)` on bool; fragile for int8.
- **Fix:** `np.where(action_mask == 1)`.

## logging_utils/csv_logger.py
- **Bug:** Wrong column schemas for step/episode/summary logs.
- **Fix:** Rewrote to spec columns; added `CSVLogger` class; separate `action_0_count`..`action_9_count`.

## explainability/explainer.py
- **Bug:** Missing episode, engagement, frustration in JSONL; incomplete reasons.
- **Fix:** Full record fields and expanded REASONS lookup.

## evaluation/metrics.py
- **Bug:** Incomplete logging fields; ablation only PPO.
- **Fix:** Full step/episode logging; ablation for all 4 RL algorithms; bandit persistent_flag passthrough.

## evaluation/plots.py
- **Bug:** Only 2 of 5 required plots; no std bands.
- **Fix:** Added convergence_comparison, action_heatmap, ablation_delta, learning_gain_by_type; fill_between std.

## main_experiment.py
- **Bug:** Missing CUDA seeds; Python 3.10 `int | None`; no console summary; PPO train path broken.
- **Fix:** `config.set_all_seeds()`; `Optional[int]`; `print_summary_table()`; fixed train routing.

## tests/
- Updated obs bound assertions; added `test_integration_smoke.py` (Phase 9).
- **Result:** 8 tests passing.

## Second Pass

Surgical API alignment with Step 1 integration script and Step 2E�2G spec items.

### environment/student_env.py
- **Bug:** No public `frustration`, `emotion_id`, `persistent_flag` accessors (only `_state` / private flag).
- **Fix:** Added property getters/setters plus read-only mirrors (`knowledge`, `engagement`, etc.).
- **Bug:** `info` missing `persistent_flag`, `last_answer_correct`, `emotion_name`, `explainer_reason`.
- **Fix:** Added spec keys (kept legacy keys for back-compat).
- **Bug:** `step()` never populated explainer text (NEW BUG 5).
- **Fix:** Calls module-level `reason_for()` after emotion update.

### reward/reward_function.py
- **Bug:** `compute_reward` only accepted `StudentState` and `last_answer_wrong` (NEW BUG 4).
- **Fix:** Polymorphic obs vectors / `StudentState`; accepts `last_answer_correct` or `last_answer_wrong`.

### fer_interface/fer_adapter.py
- **Bug:** `get_emotion` required `StudentState`, not obs vector.
- **Fix:** Coerces `np.ndarray` via `StudentState.from_vec()`.

### explainability/explainer.py
- **Bug:** `Explainer()` required `algorithm`; `explain()` took `StudentState` only.
- **Fix:** Default `algorithm="default"`; `explain()` accepts obs vector; added `reason_for()` (no file IO).
- **Bug:** `BEST_ACTION_MAP` lookup used string emotion keys.
- **Fix:** Lookup by `emotion_id` (int).

### logging_utils/csv_logger.py
- **Bug:** `CSVLogger` init required positional `run_name`; no `log_episode()` on facade.
- **Fix:** `__init__(log_dir=, algorithm=, seed=, run_name=)`; added `log_episode()` with spec kwargs.

### mdp_definition.py
- **Bug:** `BEST_ACTION_MAP` string-keyed while `config.BEST_ACTION_MAP` int-keyed (NEW BUG 6).
- **Fix:** Int-keyed map; added `BEST_ACTION_MAP_BY_NAME` for string lookups.

### evaluation/metrics.py
- **Bug:** `_adaptation_match` used string emotion; `np.std(ddof=1)` on single seed.
- **Fix:** Uses `emotion_id`; guards `len < 2` for std.

### evaluation/plots.py
- **Bug:** `pd.read_csv` without existence checks; heatmap hardcoded `config.ALGORITHMS` (NEW BUG 10).
- **Fix:** `os.path.exists` guards with warnings; dynamic algorithm list from `summary.csv`.

### main_experiment.py
- **Bug:** Missing `run_single()`, `PPO+DQN` aliases, `seed='ALL'` rows, `--parallel`.
- **Fix:** Added `AGENT_ALIASES`, `run_single()`, aggregated summary row per algorithm, spawn pool runner.

### agents/ppo_agent.py, agents/dqn_agent.py
- **Bug:** `Monitor` without `filename=` � collision risk for parallel seeds (NEW BUG 9).
- **Fix:** `filename=monitor_{algo_tag}_seed{seed}` via `algo_tag` parameter.

### __init__.py
- **Bug:** Matplotlib cache warnings in sandboxed runs.
- **Fix:** `MPLCONFIGDIR` under `~/.cache/rl_module/matplotlib`.

### tests/test_step1_integration.py
- **New:** Mirrors Step 1 inline integration script (env props, info keys, mask, emergency, cooldown, DQNSafeEnv, FER, reward, CSVLogger, Explainer).

## Third Pass

Scale-only bugs found via 3,000-step training, parallel-run inspection, and T1�T13 code audit.

### agents/bandit_dqn.py (BUG T6)
- **Bug:** `LinUCB.select()` called `np.linalg.inv(self.A[a])` with no fallback; ill-conditioned A matrices after many updates raise `LinAlgError` and crash Bandit+DQN mid-run.
- **Fix:** `try/except np.linalg.LinAlgError` with fallback `A_inv = np.eye(self.d)`.

### explainability/explainer.py (BUG T7)
- **Bug:** `explain()` wrote and flushed every call; at 50k steps � 6 algos � 10 seeds JSONL files grow to hundreds of MB with severe IO cost.
- **Fix:** `_write_count` counter; disk write only when `_write_count % 10 == 0` (~90% fewer lines). `reason_for()` in env remains no-IO.

### main_experiment.py (CLI alias)
- **Bug:** `--algo` used `choices=ALGO_ORDER` (underscore names only); `argparse` rejected `PPO+DQN` / `Bandit+DQN` before `AGENT_ALIASES` ran.
- **Fix:** Removed `choices=`; validate after `_resolve_algorithm()` with clear `parser.error()` message.

### agents/ppo_agent.py (BUG T1)
- **Bug:** `policy_kwargs = dict(net_arch=[256, 256, 256])` uses deprecated list format for SB3 ? 1.7 `MlpPolicy`.
- **Fix:** `net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])`. DQN list format unchanged.

### Verified OK (no change)
- ActionMasker + `mask_fn`; cooldowns and `consecutive_flag_steps` reset in `__init__`/`reset()`; hybrid `_ppo`/`_dqn` paths; fresh eval `StudentEnv`; int-key `BEST_ACTION_MAP`; `SummaryWriter(append=False)`; lazy pygame; `StudentEnv ? DQNSafeEnv ? Monitor`; `set_all_seeds` before population.

## Fourth Pass

Theory-grounded reward redesign (Dawes, 1979 equal weighting).

### reward/reward_function.py
- **Change:** Removed `W_KNOWLEDGE`�`W_OPTIMAL`, `BONUS_MAP`, `PENALTY_MAP`, and all action-dependent bonuses.
- **Change:** Equal weight `W = 1/7` across seven terms: knowledge gain, engagement change, confusion, boredom, frustration, zone bonus, wrong-answer penalty.
- **Change:** `action` parameter retained for API compatibility but never used in the body.
- **Change:** Sensitivity override via `config.USE_SENSITIVITY_WEIGHTS` / `config.SENSITIVITY_WEIGHTS`.

### config.py
- **Change:** Added `FRUSTRATION_HIGH_THRESHOLD`, `FRUSTRATION_PENALTY_HIGH`, `PERSISTENT_PENALTY`, `WRONG_ANSWER_PENALTY`, `ZONE_*` constants.
- **Change:** Added `USE_SENSITIVITY_WEIGHTS` and `SENSITIVITY_WEIGHTS` for sensitivity runs.
- **Change:** `BEST_ACTION_MAP` comment clarifies evaluation/explainer-only use (not reward).
- **Change:** Deleted `HYBRID_PPO_WEIGHT` and `HYBRID_DQN_WEIGHT`.
- **Change:** `BANDIT_EMOTION_DELTA_THRESHOLD` retyped to `int = 1` with citation.

### evaluation/metrics.py
- **Change:** Added `compute_adaptation_accuracy(episode_log)` � post-training metric using `config.BEST_ACTION_MAP` int keys.

### evaluation/sensitivity_analysis.py (new)
- **Change:** Weight configs (equal, knowledge_priority, affective_priority) and threshold configs (default, stricter, looser); Spearman rank correlation output.

## Fifth Pass

Principled defaults, hybrid routing, live viewer, clipping documentation.

### config.py
- **Change:** Every constant annotated with `# SOURCE:`, `# DEFAULT:`, or `# SENSITIVITY:` comment.

### agents/ppo_dqn_hybrid.py
- **Change:** Replaced fixed 0.6/0.4 weighted ensemble with `route_decision()` conditional routing (DQN on state_delta > 0.15, frustration rising, or confusion > 0.6; else PPO).
- **Change:** Exposed `last_agent_used` instance attribute (no CSV column � logging unchanged per scope).

### mdp_definition.py
- **Change:** Added CLIPPING EXPLAINED module docstring (state, reward, normalized gain); no behavior change.

### evaluation/live_viewer.py (new)
- **Change:** Separate-process pygame viewer tailing `steps_{algo}_seed{N}.csv` at 2 Hz.

### tests/test_reward_signs.py
- **Change:** Replaced action-dependent bonus test with knowledge-gain and action-invariance tests.

### README.md
- **Change:** Live viewer two-terminal usage instructions.

## Sixth Pass

Correctness audit fixes (B1-B6, T1): sensitivity overrides, threshold dedup, dead parameters.

### B1 - Threshold sensitivity was a no-op
- **Bug:** `sensitivity_analysis.py` patched `config.FRUSTRATION_BLOCK_*` only; `student_env.py` uses bare names captured from `mdp_definition` at import time.
- **Fix:** `_patch_thresholds` updates `mdp_definition`, `config`, and `student_env` module attributes together.

### B2 - Weight sensitivity silently failed
- **Bug:** `compute_reward` read only `SENSITIVITY_WEIGHTS["W"]`; `knowledge_priority` / `affective_priority` keys were ignored.
- **Fix:** Per-term overrides (`W_knowledge`, `W_engagement`, `W_affective`, etc.) and sensitivity threshold keys in `reward_function.py`.

### B3 - Duplicate masking thresholds
- **Bug:** Same constants in `mdp_definition.py` and `config.py`; env used `mdp_definition` only.
- **Fix:** `config.py` re-exports from `mdp_definition` (single source of truth).

### B4 - Incomplete factorial sweep
- **Bug:** Six separate runs (3 weights + 3 thresholds), not 3x3 combined.
- **Fix:** `run_factorial()` nested loop: 9 cells x 6 algorithms = 54 rows in `sensitivity_factorial.csv`.

### B5 - Dead `use_emotion_bonuses`
- **Bug:** Threaded through env/agents/ablation but unused after bonus maps removed.
- **Fix:** Removed from `StudentEnv`, `make_env`, `make_masked_env`, `run_ablation`, and `compute_reward`.

### B6 - Dead frustration branch
- **Bug:** `elif 0.3 < f <= HIGH` and `else` both returned `0.0`.
- **Fix:** Collapsed to single `else: frustration_term = 0.0` (productive frustration).

### T1 - Two adaptation-accuracy implementations
- **Bug:** `compute_adaptation_accuracy` used `config.BEST_ACTION_MAP` (lists); `_adaptation_match` used `mdp_definition.BEST_ACTION_MAP` (sets).
- **Fix:** `compute_adaptation_accuracy` delegates to `_adaptation_match`.
