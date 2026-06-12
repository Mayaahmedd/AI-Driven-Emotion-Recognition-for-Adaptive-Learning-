import os, json, csv, glob
from pathlib import Path

output_lines = []

def log(text=""):
    print(text)
    output_lines.append(text)

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Edit these two paths if your folders are named differently
RESULTS_DIR = Path("RESULTS")
FER_DIR     = Path("FER_Module")

LABEL_ORDER = ["boredom", "engagement", "confusion", "frustration"]

# ── HELPERS ─────────────────────────────────────────────────────────────────
def find_files(root: Path, pattern: str):
    return sorted(root.rglob(pattern))

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_csv_dicts(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def safe_float(val):
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None

# ── 1. SCAN ALL EXPERIMENT DIRECTORIES ──────────────────────────────────────
log("=" * 70)
log("EXPERIMENT DIRECTORIES FOUND")
log("=" * 70)

all_exp_dirs = []
for base in [RESULTS_DIR, FER_DIR]:
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir():
                all_exp_dirs.append(d)
                log(str(d))

log()

# ── 2. PER-EXPERIMENT METRICS ────────────────────────────────────────────────
log("=" * 70)
log("PER-EXPERIMENT METRICS")
log("=" * 70)

# Files that commonly hold final test metrics
METRIC_FILENAMES = [
    "test_metrics.json",
    "results.json",
    "metrics.json",
    "classification_report.json",
    "eval_results.json",
    "test_results.json",
]

for exp_dir in all_exp_dirs:
    found_metrics = False
    log(f"\n── {exp_dir.name}")

    # Try JSON metric files
    for fname in METRIC_FILENAMES:
        fpath = exp_dir / fname
        if fpath.exists():
            try:
                data = load_json(fpath)
                log(f"   [JSON] {fname}")
                # Print everything we find
                for k, v in data.items():
                    if isinstance(v, dict):
                        log(f"   {k}:")
                        for kk, vv in v.items():
                            log(f"      {kk}: {safe_float(vv)}")
                    else:
                        log(f"   {k}: {safe_float(v)}")
                found_metrics = True
            except Exception as e:
                log(f"   ERROR reading {fname}: {e}")

    # Try CSV metric files
    for csv_path in exp_dir.glob("*.csv"):
        try:
            rows = load_csv_dicts(csv_path)
            if rows:
                log(f"   [CSV] {csv_path.name}  ({len(rows)} rows)")
                # Print header + first few rows
                headers = list(rows[0].keys())
                log(f"   columns: {headers}")
                for row in rows[:5]:
                    log(f"   {dict(row)}")
                found_metrics = True
        except Exception as e:
            log(f"   ERROR reading {csv_path.name}: {e}")

    # Try plain text logs
    for txt_path in list(exp_dir.glob("*.txt")) + list(exp_dir.glob("*.log")):
        try:
            text = txt_path.read_text(errors="replace")
            # Look for lines with f1, precision, recall, accuracy
            hits = [
                line.strip() for line in text.splitlines()
                if any(kw in line.lower() for kw in
                       ["macro", "f1", "precision", "recall",
                        "accuracy", "boredom", "engagement",
                        "confusion", "frustration", "weighted",
                        "exact", "loss", "epoch"])
            ]
            if hits:
                log(f"   [TXT/LOG] {txt_path.name}")
                for h in hits[:40]:
                    log(f"   {h}")
                found_metrics = True
        except Exception as e:
            log(f"   ERROR reading {txt_path.name}: {e}")

    if not found_metrics:
        log("   ** NO METRIC FILES FOUND - check directory manually **")

# ── 3. TRAINING CURVES FOR EXP 38 ───────────────────────────────────────────
log()
log("=" * 70)
log("TRAINING CURVES - EXP 38 (all epoch-level data)")
log("=" * 70)

exp38_candidates = [
    d for d in all_exp_dirs
    if "exp38" in d.name.lower() or "exp_38" in d.name.lower()
]

for exp38 in exp38_candidates:
    log(f"\nDirectory: {exp38}")
    for fpath in sorted(exp38.rglob("*")):
        if fpath.is_file():
            name_lower = fpath.name.lower()
            if any(kw in name_lower for kw in
                   ["train", "val", "epoch", "curve", "history", "log"]):
                log(f"  Found: {fpath.name}")
                if fpath.suffix in [".json", ".csv", ".txt", ".log"]:
                    try:
                        text = fpath.read_text(errors="replace")
                        log(text[:3000])
                    except Exception as e:
                        log(f"  ERROR: {e}")

# ── 4. SVM BASELINES ─────────────────────────────────────────────────────────
log()
log("=" * 70)
log("SVM / TRADITIONAL BASELINES")
log("=" * 70)

svm_candidates = [
    d for d in all_exp_dirs
    if any(kw in d.name.lower() for kw in ["svm", "hog", "lbp", "baseline"])
]

if not svm_candidates:
    log("No SVM directories found by name. Searching all dirs for SVM keywords...")
    for exp_dir in all_exp_dirs:
        for fpath in exp_dir.rglob("*"):
            if fpath.is_file() and fpath.suffix in [".json", ".csv", ".txt", ".log"]:
                try:
                    text = fpath.read_text(errors="replace")
                    if "svm" in text.lower() or "hog" in text.lower():
                        log(f"  Possible SVM result in: {fpath}")
                        hits = [l.strip() for l in text.splitlines()
                                if any(k in l.lower() for k in
                                       ["precision","recall","f1","accuracy"])]
                        for h in hits[:20]:
                            log(f"    {h}")
                except Exception:
                    pass
else:
    for d in svm_candidates:
        log(f"\n{d}")
        for fpath in sorted(d.rglob("*")):
            if fpath.is_file():
                log(f"  {fpath.name}")

# ── 5. DAISEE CLASS DISTRIBUTION ─────────────────────────────────────────────
log()
log("=" * 70)
log("DAiSEE CLASS DISTRIBUTION (test set)")
log("=" * 70)

dist_candidates = []
for base in [RESULTS_DIR, FER_DIR, Path(".")]:
    dist_candidates += list(base.rglob("*class_dist*"))
    dist_candidates += list(base.rglob("*label_dist*"))
    dist_candidates += list(base.rglob("*distribution*"))
    dist_candidates += list(base.rglob("*daisee_stats*"))

if dist_candidates:
    for f in dist_candidates:
        log(f"  Found: {f}")
        try:
            log(f.read_text(errors="replace")[:1000])
        except Exception:
            pass
else:
    log("  No distribution file found.")
    log("  Run this separately to get test set class counts:")
    log()
    log("  import pandas as pd")
    log("  df = pd.read_csv('your_test_labels.csv')")
    log("  print(df[['boredom','engagement','confusion','frustration']].sum())")
    log("  print(df[['boredom','engagement','confusion','frustration']].mean())")

# ── 6. OVERSAMPLING CONFIG FOR EXP 16 AND 17 ─────────────────────────────────
log()
log("=" * 70)
log("OVERSAMPLING CONFIG - EXP 16 AND EXP 17")
log("=" * 70)

for exp_id in ["exp16", "exp17", "exp_16", "exp_17"]:
    candidates = [d for d in all_exp_dirs if exp_id in d.name.lower()]
    for d in candidates:
        log(f"\n{d}")
        for fpath in sorted(d.rglob("*")):
            if fpath.is_file() and fpath.suffix in \
               [".json", ".yaml", ".yml", ".py", ".cfg", ".txt", ".log"]:
                name_lower = fpath.name.lower()
                if any(kw in name_lower for kw in
                       ["config", "param", "setting", "oversamp",
                        "train", "run", "args", "log"]):
                    log(f"  {fpath.name}")
                    try:
                        text = fpath.read_text(errors="replace")
                        hits = [l.strip() for l in text.splitlines()
                                if any(k in l.lower() for k in
                                       ["oversamp", "multiplier", "weight",
                                        "sampl", "ratio", "boost"])]
                        for h in hits[:20]:
                            log(f"    {h}")
                    except Exception:
                        pass

# ── 7. UNACCOUNTED EXPERIMENTS ───────────────────────────────────────────────
log()
log("=" * 70)
log("UNACCOUNTED EXPERIMENTS (7, 20, 21, 29, 32, 33, 39, 40, 41)")
log("=" * 70)

for exp_id in ["exp7","exp20","exp21","exp29","exp32",
               "exp33","exp39","exp40","exp41"]:
    candidates = [d for d in all_exp_dirs if exp_id in d.name.lower()]
    if candidates:
        for d in candidates:
            log(f"  FOUND: {d}")
    else:
        log(f"  NOT FOUND: {exp_id}")

# ── SAVE OUTPUT ───────────────────────────────────────────────────────────────
out_path = Path("thesis_data_extracted.txt")
out_path.write_text("\n".join(output_lines), encoding="utf-8")
print(f"\n\nSaved to: {out_path.resolve()}")