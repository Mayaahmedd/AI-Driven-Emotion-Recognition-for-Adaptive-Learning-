"""
Generate the eight FER figures referenced in chapters/results.tex Part I.

Run from the Bachelor_thesis_MET_Bachelor/ directory:
    python Figures/generate_fer_figures.py

All numbers are hard-coded from the verified data sheet so the script
is reproducible without external CSV files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# Pastel palette
# ----------------------------------------------------------------------
PASTEL_BLUE   = "#A8C5DA"
PASTEL_GREEN  = "#A8D5B5"
PASTEL_CORAL  = "#F2A99E"
PASTEL_YELLOW = "#F5DFA0"
PASTEL_PURPLE = "#C3B1D6"
PASTEL_TEAL   = "#A0CEC8"
PASTEL_PEACH  = "#F7C9A3"
ACCENT_BLUE   = "#7BAEC8"

# ----------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titlesize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"wrote {path}")
    plt.close(fig)


# ======================================================================
# Figure 1: Multi-label Macro F1 across all experiments
# ======================================================================
def figure1_macro_f1_all():
    rows = [
        # (label, value, family)
        ("Exp 6  ResNet50+BiLSTM (WRS, fail)",          0.153, "A"),
        ("Exp 13 Xception+BiLSTM",                       0.246, "D"),
        ("Exp 1  ResNet50+BiLSTM (BCE)",                 0.251, "A"),
        ("Exp 8  ResNet50+BiLSTM (last frame)",          0.303, "A"),
        ("Exp 15 ResNet18+Transformer",                  0.305, "E"),
        ("Exp 11 EfficientNetV2-S+Transformer",          0.319, "B"),
        ("Exp 2  ResNet50+BiLSTM (WBCE)",                0.356, "A"),
        ("Exp 19 3D DenseNet+Self-Attn",                 0.379, "F"),
        ("Exp 10 EfficientNetV2-S+BiLSTM",               0.379, "B"),
        ("Exp 34 ResNet50+BiLSTM+Attn+OS",               0.383, "A"),
        ("Exp 3  ResNet50+BiLSTM (thr=0.3)",             0.392, "A"),
        ("Exp 12 ShuffleNetV2+SE+BiLSTM",                0.404, "C"),
        ("Exp 5  ResNet50+BiLSTM (lr sched)",            0.405, "A"),
        ("Exp 4  ResNet50+BiLSTM (5 ep, WBCE)",          0.408, "A"),
        ("Exp 16 ShuffleNetV2+SE+2BiLSTM (WRS)",         0.445, "C"),
        ("Exp 38 ResNet50+SE+2BiLSTM+Attn (final)",      0.453, "H"),
    ]
    family_colour = {
        "A": PASTEL_BLUE,
        "B": PASTEL_GREEN,
        "C": PASTEL_CORAL,
        "D": PASTEL_YELLOW,
        "E": PASTEL_PURPLE,
        "F": PASTEL_TEAL,
        "H": ACCENT_BLUE,
    }
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colours = [family_colour[r[2]] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colours, edgecolor="#333333", linewidth=0.5)
    # Highlight final model
    bars[-1].set_edgecolor("#333333")
    bars[-1].set_linewidth(1.4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Macro F1")
    ax.set_xlim(0, 0.55)
    ax.set_title("Multi-label Macro F1 across all experiments")
    ax.grid(axis="x", linestyle=":", color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)

    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    legend_handles = [
        Patch(facecolor=family_colour[k], edgecolor="#333333", label=f"Family {k}")
        for k in ["A", "B", "C", "D", "E", "F", "H"]
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)

    fig.tight_layout()
    save(fig, "fer_macro_f1_all.png")


# ======================================================================
# Figure 2: Per-label precision, recall, F1 for Exp 38
# ======================================================================
def figure2_per_label_metrics():
    labels = ["Boredom", "Engagement", "Confusion", "Frustration"]
    precision = [0.239, 0.951, 0.171, 0.205]
    recall    = [0.708, 1.000, 0.325, 0.338]
    f1        = [0.358, 0.975, 0.224, 0.255]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, precision, width, color=PASTEL_BLUE,  label="Precision",
           edgecolor="#333333", linewidth=0.5)
    ax.bar(x,         recall,    width, color=PASTEL_GREEN, label="Recall",
           edgecolor="#333333", linewidth=0.5)
    ax.bar(x + width, f1,        width, color=PASTEL_PEACH, label="F1",
           edgecolor="#333333", linewidth=0.5)

    for xi, val in zip(x - width, precision):
        ax.text(xi, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    for xi, val in zip(x, recall):
        ax.text(xi, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    for xi, val in zip(x + width, f1):
        ax.text(xi, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-label precision, recall, and F1 of the final FER model (Exp 38)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fer_per_label_metrics.png")


# ======================================================================
# Figure 3: Overall metrics for Exp 38
# ======================================================================
def figure3_overall_metrics():
    metrics = ["Macro F1", "Micro F1", "Weighted F1", "Samples F1",
               "Exact Match", "Label Acc."]
    values = [0.453, 0.724, 0.798, 0.746, 0.342, 0.782]
    colours = [PASTEL_CORAL, PASTEL_BLUE, PASTEL_GREEN,
               PASTEL_PEACH, PASTEL_PURPLE, PASTEL_TEAL]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    ax.bar(x, values, color=colours, edgecolor="#333333", linewidth=0.5)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Overall test metrics of the final FER model (Exp 38)")
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fer_overall_metrics.png")


# ======================================================================
# Figure 4: Per-label accuracy
# ======================================================================
def figure4_per_label_accuracy():
    labels = ["Boredom", "Engagement", "Confusion", "Frustration"]
    accuracy = [0.462, 0.951, 0.802, 0.911]
    colours = [PASTEL_PEACH, PASTEL_GREEN, PASTEL_BLUE, PASTEL_CORAL]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, accuracy, color=colours, edgecolor="#333333", linewidth=0.5)
    for xi, v in zip(x, accuracy):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)

    ax.axhline(0.5, linestyle="--", color="#888888", linewidth=1.0)
    ax.text(len(labels) - 0.5, 0.515, "chance baseline", fontsize=9,
            color="#666666", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Per-label accuracy")
    ax.set_title("Per-label accuracy of the final FER model (Exp 38)")
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fer_per_label_accuracy.png")


# ======================================================================
# Figure 5: Training loss and validation Macro F1 across epochs
# ======================================================================
def figure5_training_curves():
    epochs = [1, 2, 3, 4, 5]
    train_loss = [0.7605, 0.6786, 0.6313, 0.5810, 0.4938]
    val_macro  = [0.5117, 0.4986, 0.5000, 0.4977, 0.4795]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    line1, = ax1.plot(epochs, train_loss, marker="o", color=PASTEL_CORAL,
                      linewidth=2, label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="#7a4944")
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis="y", labelcolor="#7a4944")
    ax1.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    line2, = ax2.plot(epochs, val_macro, marker="s", color=PASTEL_BLUE,
                      linewidth=2, label="Validation Macro F1")
    ax2.set_ylabel("Validation Macro F1", color="#3f6c89")
    ax2.set_ylim(0, 0.7)
    ax2.tick_params(axis="y", labelcolor="#3f6c89")

    ax1.axvline(1, linestyle="--", color=PASTEL_PURPLE, linewidth=1.5)
    ax1.text(1.05, 0.95, "early stopping\ncheckpoint", color="#5e4a7a",
             fontsize=9, va="top")

    ax1.set_xticks(epochs)
    ax1.set_title("Training loss and validation Macro F1 across epochs")
    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right",
               frameon=False)

    fig.tight_layout()
    save(fig, "fer_training_curves.png")


# ======================================================================
# Figure 6: Single-label vs multi-label Macro F1 for matched architectures
# ======================================================================
def figure6_single_vs_multi():
    archs = ["HOG+LBP+SVM", "ResNet50+SVM",
             "ResNet50\n+BiLSTM+Attn",
             "ShuffleNetV2\n+BiLSTM+Attn",
             "ResNet50\n+SE+2BiLSTM+Attn",
             "ShuffleNetV2\n+SE+2BiLSTM+Attn"]
    single = [0.214, 0.229, 0.311, 0.266, 0.271, 0.269]
    multi  = [np.nan, np.nan, 0.383, np.nan, 0.453, 0.445]
    final_mask = [False, False, False, False, True, False]

    x = np.arange(len(archs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars_single = ax.bar(x - width / 2, single, width, color=PASTEL_PEACH,
                         edgecolor="#333333", linewidth=0.5,
                         label="Single-label Macro F1")
    multi_colours = [ACCENT_BLUE if m else PASTEL_BLUE for m in final_mask]
    multi_plot = [v if not np.isnan(v) else 0 for v in multi]
    bars_multi = ax.bar(x + width / 2, multi_plot, width, color=multi_colours,
                        edgecolor="#333333", linewidth=0.5,
                        label="Multi-label Macro F1")
    # Hide the zero-height bars for n/a cells
    for i, v in enumerate(multi):
        if np.isnan(v):
            bars_multi[i].set_visible(False)
            ax.text(x[i] + width / 2, 0.01, "n/a", ha="center",
                    fontsize=9, color="#777777")

    for xi, v in zip(x - width / 2, single):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + width / 2, multi):
        if not np.isnan(v):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=9)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 0.55)
    ax.set_title("Single-label vs multi-label Macro F1 for matched architectures")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(len(archs) - 1, 0.52,
            "Note: metrics are not directly comparable across formulations",
            ha="right", fontsize=8, color="#666666", style="italic")

    fig.tight_layout()
    save(fig, "fer_single_vs_multi.png")


# ======================================================================
# Figure 7: Family A imbalance-intervention progression
# ======================================================================
def figure7_familyA_progression():
    steps = [
        "BCE, thr 0.5\n(Exp 1)",
        "WBCE, thr 0.5\n(Exp 2)",
        "WBCE, thr 0.3\n(Exp 3)",
        "WBCE, 5 epochs\n(Exp 4)",
        "WRS+WBCE\n(Exp 6, failure)",
        "OS+WBCE, thr 0.5\n(Exp 34)",
        "OS+WBCE, tuned thr\n(Exp 38, final)",
    ]
    values = [0.251, 0.356, 0.392, 0.408, 0.153, 0.383, 0.453]
    # Gradient colour by value, with Exp 6 highlighted in coral
    colours = [PASTEL_YELLOW, PASTEL_BLUE, PASTEL_BLUE, PASTEL_BLUE,
               PASTEL_CORAL, PASTEL_GREEN, ACCENT_BLUE]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(steps))
    bars = ax.bar(x, values, color=colours, edgecolor="#333333", linewidth=0.5)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # Annotation arrow at Exp 6
    ax.annotate("double-correction\nfailure",
                xy=(4, 0.153), xytext=(4, 0.30),
                ha="center", fontsize=9, color="#8b3a30",
                arrowprops=dict(arrowstyle="->", color="#8b3a30", linewidth=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=9)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Macro F1")
    ax.set_title("Effect of imbalance interventions on Macro F1 (Family A progression)")
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fer_familyA_progression.png")


# ======================================================================
# Figure 8: SVM baselines vs Exp 38 per-label F1
# ======================================================================
def figure8_svm_vs_final():
    models = ["HOG+LBP+SVM", "ResNet50+SVM", "Exp 38 (final)"]
    boredom     = [0.000, 0.000, 0.358]
    engagement  = [0.857, 0.863, 0.975]
    confusion   = [0.000, 0.000, 0.224]
    frustration = [0.000, 0.053, 0.255]

    x = np.arange(len(models))
    width = 0.20

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 1.5 * width, boredom,     width, color=PASTEL_PEACH,
           edgecolor="#333333", linewidth=0.5, label="Boredom F1")
    ax.bar(x - 0.5 * width, engagement,  width, color=PASTEL_GREEN,
           edgecolor="#333333", linewidth=0.5, label="Engagement F1")
    ax.bar(x + 0.5 * width, confusion,   width, color=PASTEL_BLUE,
           edgecolor="#333333", linewidth=0.5, label="Confusion F1")
    ax.bar(x + 1.5 * width, frustration, width, color=PASTEL_CORAL,
           edgecolor="#333333", linewidth=0.5, label="Frustration F1")

    for xi, v in zip(x - 1.5 * width, boredom):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x - 0.5 * width, engagement):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.5 * width, confusion):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + 1.5 * width, frustration):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Per-label F1")
    ax.set_title("Per-label F1: SVM baselines vs final multi-label model")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fer_svm_vs_final.png")


if __name__ == "__main__":
    figure1_macro_f1_all()
    figure2_per_label_metrics()
    figure3_overall_metrics()
    figure4_per_label_accuracy()
    figure5_training_curves()
    figure6_single_vs_multi()
    figure7_familyA_progression()
    figure8_svm_vs_final()
    print("All eight figures regenerated.")
