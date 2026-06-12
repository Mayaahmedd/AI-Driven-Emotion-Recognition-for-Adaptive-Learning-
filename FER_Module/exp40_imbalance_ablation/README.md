# Exp40 — Imbalance Correction Ablation

A controlled ablation isolating the effect of imbalance-handling strategies on multilabel DAiSEE FER. Three variants share the same architecture (ResNet50 + SE + 2-layer BiLSTM + Temporal Attention + multilabel head), the same data pipeline (exp38-style per-frame augmentation, T = 24 frames, ImageNet normalisation, 224×224), the same optimiser, scheduler, batch size, early-stopping rule, and threshold-tuning grid. Only the imbalance-correction switches differ.

| Variant | use_oversampling | use_pos_weight | Description |
|---|:---:|:---:|---|
| **A** | ✅ | ❌ | Data-level only. BCE `pos_weight = ones(4)`. |
| **B** | ❌ | ✅ | Loss-level only. Trains on the original DAiSEE `train_multilabel.csv`. |
| **C** | ✅ | ✅ | Combined (matches the exp38 baseline configuration). |

Threshold tuning on validation is active in all three variants. This is *not* an imbalance corrector — it is the decision rule applied uniformly to every variant, so it stays inside the ablation rather than being one of its dimensions.

The original exp38 and exp39 directories are not modified. This experiment is fully self-contained under `FER_Module/exp40_imbalance_ablation/`.

## Layout

```
exp40_imbalance_ablation/
├── README.md
└── src_exp40/
    ├── __init__.py
    ├── dataset_exp40.py             ← byte-identical to exp38 dataset
    ├── model_exp40.py               ← same architecture as exp38, renamed class
    ├── make_balanced_csv_exp40.py   ← builds the oversampled CSV (for variants A, C)
    ├── train_exp40.py               ← single-variant trainer (config-driven)
    ├── evaluate_exp40.py            ← single-variant test-set evaluation
    ├── run_all_variants.py          ← orchestrator (A → B → C → compare)
    ├── compare_variants.py          ← builds the final A vs B vs C report
    └── configs/
        ├── variant_A_oversampling_only.json
        ├── variant_B_posweight_only.json
        └── variant_C_combined.json
```

## Kaggle execution

Upload `src_exp40/` to `/kaggle/working/FER_Project/src_exp40` and run:

```bash
# end-to-end: build CSV → train A,B,C → evaluate A,B,C → comparison report
python /kaggle/working/FER_Project/src_exp40/run_all_variants.py
```

Or run one variant at a time:

```bash
python /kaggle/working/FER_Project/src_exp40/make_balanced_csv_exp40.py
python /kaggle/working/FER_Project/src_exp40/train_exp40.py --config /kaggle/working/FER_Project/src_exp40/configs/variant_A_oversampling_only.json
python /kaggle/working/FER_Project/src_exp40/train_exp40.py --config /kaggle/working/FER_Project/src_exp40/configs/variant_B_posweight_only.json
python /kaggle/working/FER_Project/src_exp40/train_exp40.py --config /kaggle/working/FER_Project/src_exp40/configs/variant_C_combined.json
python /kaggle/working/FER_Project/src_exp40/evaluate_exp40.py --variant A
python /kaggle/working/FER_Project/src_exp40/evaluate_exp40.py --variant B
python /kaggle/working/FER_Project/src_exp40/evaluate_exp40.py --variant C
python /kaggle/working/FER_Project/src_exp40/compare_variants.py
```

## What is kept constant across variants

* Architecture: ResNet50 + SE block + 2-layer BiLSTM + Temporal Attention + MLP head.
* Frame sampling: `T = 24` via `np.linspace(0, total-1, 24).astype(int)`.
* Spatial pipeline: BGR → RGB, resize 224×224, ImageNet mean/std normalisation.
* Augmentation (per-frame, as in exp38): flip 0.5, brightness/contrast 0.5, rotation 0.25, blur 0.15, crop 0.25.
* Loss: `BCEWithLogitsLoss` (with or without `pos_weight` depending on variant).
* Optimiser: AdamW, lr = 5e-5, weight decay = 1e-5, grad-clip = 5.0.
* Scheduler: `ReduceLROnPlateau` on validation Macro F1, factor 0.5, patience 2.
* Training budget: 12 epochs, early-stop patience = 4 on validation Macro F1.
* Batch size: 4. Threshold grid: 0.10 → 0.70 step 0.05. Seed: 42.

## What varies between variants

| Knob | Variant A | Variant B | Variant C |
|---|---|---|---|
| Training CSV | `processed_exp40/train_multilabel_exp40_oversampled.csv` | `…/train_multilabel.csv` (original) | `processed_exp40/train_multilabel_exp40_oversampled.csv` |
| BCE `pos_weight` | `ones(4)` | `clip(neg/pos, 0.5, 6.0)` from the original training CSV | `clip(neg/pos, 0.5, 6.0)` from the oversampled CSV |

Nothing else is changed.

## Required outputs

After `run_all_variants.py` completes, `checkpoints/exp40_imbalance_ablation/` contains:

```
variant_A_oversampling_only/
    best_model.pt
    best_val_report.json
    train_log.json
    test_report.json
    test_report.txt
    test_predictions.csv
    config.json
variant_B_posweight_only/    (same files)
variant_C_combined/          (same files)
comparison_summary.json
comparison_summary_val.csv
comparison_summary_test.csv
comparison_summary.md        ← the A vs B vs C table + recommendation
```

For each variant the test report contains:

* Macro F1, Micro F1, Weighted F1, Samples F1.
* Exact-match accuracy.
* Per-label F1, precision, recall, accuracy.
* **Per-label confusion-style summary** (TP, FP, TN, FN, support_positive, support_negative).
* sklearn classification report.

## Recommendation rule

`compare_variants.py` picks the variant with the highest **validation** Macro F1. If two or more variants are within a tie band of 0.005 (≈ 0.5 F1 points, which is the noise floor for single-seed DAiSEE runs), the simpler imbalance correction is preferred (A or B before C). This rule is conservative: it rewards combined correction only when it produces a non-trivial improvement.

## Forbidden additions (kept out by design)

This is a *pure* imbalance ablation. The following techniques would muddy the comparison and are intentionally not included:

* EMA of weights.
* MixUp / CutMix.
* Label smoothing.
* Cosine / warm-restart LR schedules.
* Architecture modifications (focal loss, CB loss, asymmetric loss, etc.).
* Test-time augmentation or ensembling.

If any of these is needed, it belongs in a separate, clearly-named follow-up experiment so that A/B/C remain interpretable as imbalance-correction comparisons.
