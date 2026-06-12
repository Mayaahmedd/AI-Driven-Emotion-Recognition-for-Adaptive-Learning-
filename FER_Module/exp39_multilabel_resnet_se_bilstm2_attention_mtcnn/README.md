# Exp39 — Multilabel DAiSEE FER with Per-Clip Augmentation and MTCNN Face Cropping

This experiment is a controlled refactor of `exp38_multilabel_resnet_se_bilstm2_attention_24frames`. The model architecture and every training hyperparameter are preserved exactly; only the **data pipeline** is changed, so any Macro-F1 difference between the two experiments is attributable to the data-pipeline changes alone.

The original exp38 directory is **not modified**. It remains the control condition.

## What changed relative to exp38

| Area | exp38 (baseline) | exp39 (this experiment) |
|---|---|---|
| Augmentation sampling | One random draw **per frame** inside `_augment` | One random draw **per clip**; the same flip / brightness-contrast / rotation / blur / crop tuple is applied identically to every frame in the 24-frame sequence |
| Spatial crop | None — whole 224x224 webcam frame | MTCNN detects the largest visible face on the anchor frame; bounding box is expanded by `face_margin=0.20`, clipped to image bounds, and applied to every frame in the clip. Falls back to the full frame if MTCNN finds no face. |
| Face-detector compute cost | n/a | Bounding boxes cached on disk (`processed_exp39/face_bboxes.json`), so each clip is processed by MTCNN exactly once across all epochs and across train + test |
| Frame-IO failure handling | Silent zero-frame substitution on `cv2.imread is None` | Warning logged; the previous successfully-loaded frame is reused; zero-fill only if no prior frame is available |
| Augmentation RNG | Unseeded `numpy.random` | Per-worker `numpy.random.Generator` derived deterministically from a single `SEED` constant via `worker_init_fn` |
| ImageNet weights variant | `ResNet50_Weights.DEFAULT` (silently V2) | `ResNet50_Weights.IMAGENET1K_V2` named explicitly |

Everything else is identical to exp38:

- Label set `["Boredom", "Engagement", "Confusion", "Frustration"]` with binary 0/1 values produced upstream by the Malekshahi-et-al. ordinal-to-binary mapping (`levels {0,1} → 0`, `levels {2,3} → 1`).
- `T = 24` frame sampling rule (`np.linspace(0, total-1, 24).astype(int)`).
- 224x224 input, ImageNet RGB normalisation, BGR→RGB.
- Dominant-group oversampling with multipliers `{Engagement:1, Boredom:2, Confusion:3, Frustration:4, None:1}` and shuffle seed 42.
- `BCEWithLogitsLoss` with `pos_weight` per label, clipped to `[0.5, 6.0]`.
- AdamW, `lr=5e-5`, `weight_decay=1e-5`, batch size 4, grad clip 5.0.
- `ReduceLROnPlateau` on validation Macro F1 with `factor=0.5`, `patience=2`.
- 12 epochs, early-stopping patience 4 epochs on Macro F1.
- Per-label threshold grid search on validation `{0.10, 0.15, …, 0.70}`.
- Test-time evaluation uses thresholds saved with the best-Macro-F1 checkpoint; the test set is touched only once.

## File layout

```
exp39_multilabel_resnet_se_bilstm2_attention_mtcnn/
├── README.md                        (this file)
├── requirements_exp39.txt           (adds facenet-pytorch)
├── src_exp39/
│   ├── __init__.py
│   ├── face_detector_exp39.py       (MTCNN wrapper + bbox cache)
│   ├── dataset_exp39.py             (per-clip aug + face crop)
│   ├── model_exp39.py               (same architecture as exp38, renamed class)
│   ├── make_balanced_csv_exp39.py   (oversampling, writes to processed_exp39)
│   ├── train_exp39.py               (deterministic training loop)
│   └── evaluate_exp39.py            (test-set evaluation)
└── processed_exp39/                 (created at runtime: oversampled CSV + bbox cache)
```

## Kaggle execution

Upload the `src_exp39/` folder to `/kaggle/working/FER_Project/src_exp39` and run, in order:

```bash
pip install -r /kaggle/working/FER_Project/exp39_multilabel_resnet_se_bilstm2_attention_mtcnn/requirements_exp39.txt
python /kaggle/working/FER_Project/src_exp39/make_balanced_csv_exp39.py
python /kaggle/working/FER_Project/src_exp39/train_exp39.py
python /kaggle/working/FER_Project/src_exp39/evaluate_exp39.py
```

The DAiSEE frames are expected at `/kaggle/input/datasets/mayaahmedd59/daisee-farmes/frames/frames/` (identical to exp38).

## Reproducibility notes

* The single source of seed is `SEED = 42` in `train_exp39.py`. It seeds Python, NumPy, PyTorch, CUDA, the DataLoader generator and (via `worker_init_fn`) every worker's NumPy RNG.
* cuDNN benchmarking is disabled and cuDNN deterministic mode is enabled. Strict `torch.use_deterministic_algorithms(True)` is **not** turned on because bidirectional LSTM CUDA kernels require non-deterministic paths on some CUDA versions; this matches the exp38 baseline.
* The MTCNN detector is fully deterministic given the input frame; the bbox cache makes a re-run produce byte-identical crops.

## Direct A/B comparison with exp38

Both experiments use the same DAiSEE splits, the same oversampled training-set composition, the same backbone, and the same hyperparameters. Reporting both their Macro-F1, per-label F1, and exact-match accuracy in Chapter 5 isolates the contribution of (a) per-clip augmentation and (b) MTCNN face cropping. No other variable changes.
