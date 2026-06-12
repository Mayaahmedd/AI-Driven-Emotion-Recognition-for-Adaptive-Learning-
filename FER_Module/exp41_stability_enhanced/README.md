# Exp41 — Stability-Enhanced Training

This experiment isolates **training-stability** improvements over the
exp40-C baseline (= the exp38 best configuration). The architecture, the
data pipeline, the loss function, the imbalance-correction strategy, the
threshold-tuning protocol and every other knob are held fixed at the
exp40-C settings. Only three things change, and they are all training-time
interventions:

1. **EMA of weights** with decay = 0.999, updated every iteration.
2. **Cosine annealing LR with a 1-epoch linear warm-up**, stepped per
   iteration. Replaces `ReduceLROnPlateau`.
3. **Top-3 best-EMA checkpoint tracking**, enabling an optional
   state-dict-averaged ensemble at test time.

Original `exp38`, `exp39`, and `exp40` experiments are not touched.

## Layout

```
exp41_stability_enhanced/
├── README.md
└── src_exp41/
    ├── __init__.py
    ├── dataset_exp41.py            ← byte-identical to exp38/exp40 dataset
    ├── model_exp41.py              ← same architecture as exp38 (class renamed)
    ├── make_balanced_csv_exp41.py  ← oversampling rule from exp38
    ├── ema.py                      ← ModelEMA wrapper, decay 0.999
    ├── scheduler.py                ← cosine + 1-epoch linear warm-up (per-iter)
    ├── train_exp41.py              ← trains live + EMA, tracks top-3
    ├── evaluate_exp41.py           ← supports best_ema / best_live / topk_ensemble
    └── compare_to_exp40c.py        ← builds the exp41 vs exp40-C comparison
```

## Kaggle execution

```bash
# 1. build the oversampled CSV (skip if already produced)
python /kaggle/working/FER_Project/src_exp41/make_balanced_csv_exp41.py

# 2. train: writes
#    - best_model_ema.pt           (single best EMA checkpoint)
#    - best_model_live.pt          (single best LIVE checkpoint)
#    - best_ema_top1.pt … top3.pt  (top-3 best EMA checkpoints for ensembling)
#    - train_log.json              (per-epoch LIVE vs EMA validation metrics)
python /kaggle/working/FER_Project/src_exp41/train_exp41.py

# 3. evaluate (run all three modes)
python /kaggle/working/FER_Project/src_exp41/evaluate_exp41.py --mode best_ema
python /kaggle/working/FER_Project/src_exp41/evaluate_exp41.py --mode best_live
python /kaggle/working/FER_Project/src_exp41/evaluate_exp41.py --mode topk_ensemble

# 4. build the comparison report against exp40-C
python /kaggle/working/FER_Project/src_exp41/compare_to_exp40c.py
```

## What is held constant (vs exp40-C)

| Item | exp40-C | exp41 |
|---|---|---|
| Architecture | ResNet50 + SE + 2-layer BiLSTM + Temporal Attention + MLP head | **same** |
| Frame sampling | T = 24 via `np.linspace(0, total-1, 24).astype(int)` | **same** |
| Spatial pipeline | BGR→RGB, resize 224, ImageNet mean/std | **same** |
| Augmentation | exp38 per-frame policy (flip/brightness-contrast/rotation/blur/crop) | **same** |
| Loss | `BCEWithLogitsLoss` with `pos_weight` clipped to [0.5, 6.0] | **same** |
| Imbalance | oversampling + `pos_weight` (combined) | **same** |
| Threshold tuning | per-label grid 0.10 → 0.70 step 0.05 on validation | **same** |
| Optimiser | AdamW, lr 5e-5, wd 1e-5, grad-clip 5.0 | **same** |
| Batch size | 4 | **same** |
| Epochs | 12 (early-stop patience 4) | **same** |
| Seed | 42 (Python, NumPy, PyTorch, CUDA, DataLoader worker) | **same** |

## What changes (vs exp40-C)

| Item | exp40-C | exp41 |
|---|---|---|
| LR schedule | `ReduceLROnPlateau` on val Macro F1 | **Cosine annealing with 1-epoch linear warm-up, stepped per iteration** |
| Weight averaging | none | **EMA with decay 0.999, updated every iteration** |
| Eval-time model | best LIVE checkpoint | **best EMA checkpoint (default); optional top-3 EMA ensemble** |
| Early-stopping signal | val Macro F1 on live weights | **val Macro F1 on EMA weights** |
| Per-epoch logging | live metrics only | **both LIVE and EMA validation metrics every epoch** |

## What is **not** changed (forbidden by the brief, and verified absent)

- ❌ Mixup / CutMix
- ❌ Label smoothing
- ❌ Architecture changes (no focal head, no new attention, no input fusion)
- ❌ New losses (no focal loss, no CB loss, no asymmetric loss)
- ❌ New augmentation policies (the augment list is byte-identical to exp38)

## Per-mode outputs

For each evaluation mode (`best_ema`, `best_live`, `topk_ensemble`) the
script writes:

```
checkpoints/exp41_stability_enhanced/
    test_report_<mode>.json
    test_report_<mode>.txt
    test_predictions_<mode>.csv
```

Each report contains: Macro F1, Micro F1, Weighted F1, Samples F1, exact
match, per-label F1 + precision + recall, and a per-label confusion-style
summary (TP / FP / TN / FN, supports).

## Comparison report

`compare_to_exp40c.py` produces:

```
checkpoints/exp41_stability_enhanced/
    comparison_exp41_vs_exp40c.json
    comparison_exp41_vs_exp40c.md   ← human-readable
```

The markdown report contains three tables:

1. **Test-set summary** — Macro/Micro/Weighted/Samples F1 + exact match
   for exp40-C and every exp41 mode that has been evaluated.
2. **Per-label F1 (test)** — for exp40-C and every exp41 mode.
3. **Per-epoch EMA vs non-EMA validation** — read from
   `train_log.json`, this is the second deliverable required by the
   brief ("EMA vs non-EMA performance comparison"). It also lists the
   best LIVE and best EMA validation Macro F1.

## Notes on the optional top-3 ensemble

Because every saved checkpoint shares the exact same architecture, the
ensemble is constructed by **state-dict averaging** (Polyak-style), not
by output averaging. This is O(1) at inference time — only one forward
pass is needed, on the averaged weights — and is mathematically valid
because all checkpoints originate from the same training trajectory of
EMA weights. The thresholds attached to the rank-1 (best) checkpoint are
reused for the ensemble.
