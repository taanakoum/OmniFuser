"""The README's figures, in a light and a dark version.

The numbers come from the reports (FINAL_REPORT_REPRODUCTION.md, FIXES_AND_VALIDATION.md),
which derive them from experiment_results/. Usage, from the repository root:

    python scripts/readme_figures.py            # writes docs/readme/*-{light,dark}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/readme")
OUT.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": {"text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "series": ["#2a78d6", "#eb6834", "#1baf7a"],
                  "neutral": "#8a8984", "band": "#efeeea", "seq": "Blues"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "grid": "#3a3a37", "series": ["#3987e5", "#d95926", "#199e70"],
                 "neutral": "#8f8e88", "band": "#2a2a28", "seq": "Blues"},
}


def style(ax, t, grid_axis="y"):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=10)
    ax.yaxis.label.set_color(t["muted"])
    ax.xaxis.label.set_color(t["muted"])
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name, mode):
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=170, transparent=True, bbox_inches="tight")
    plt.close(fig)


for mode, t in THEMES.items():
    blue, orange, aqua = t["series"]

    # 1. Forecast error over the horizon
    h = [12, 24, 36, 48, 60, 72, 84, 96]
    series = [
        ("Ours, all tools", [1.034, 0.842, 0.779, 0.761, 0.764, 0.762, 0.782, 0.809], blue, "-"),
        ("Ours without T1, T5", [0.493, 0.505, 0.500, 0.504, 0.521, 0.525, 0.544, 0.559], blue, "--"),
        ("Paper (OmniFuser)", [0.467, 0.514, 0.586, 0.686, 0.772, 0.830, 0.917, 0.902], orange, "-"),
        ("Last-value baseline", [0.669, 0.654, 0.658, 0.682, 0.720, 0.752, 0.800, 0.867], t["neutral"], ":"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    style(ax, t)
    for label, values, color, dash in series:
        ax.plot(h, values, dash, color=color, linewidth=2.2, marker="o", markersize=5, label=label)
        ax.text(97.5, values[-1], label, color=t["text"], fontsize=9, va="center")
    ax.set_xlim(8, 128)
    ax.set_xticks(h)
    ax.set_xlabel("Forecast horizon h (seconds)")
    ax.set_ylabel("Cumulative MSE, z-scored")
    ax.set_title("Forecast error matches the paper's level; the shape differs for two measured reasons",
                 color=t["text"], fontsize=11.5, loc="left", pad=12)
    save(fig, "forecast_horizon", mode)

    # 2. Confusion matrix (row-normalized), main run
    counts = np.array([[11334, 1411, 709], [2408, 5556, 3441], [671, 2850, 7044]])
    share = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.set_facecolor("none")
    ax.imshow(share, cmap=t["seq"], vmin=0, vmax=1)
    names = ["sharp", "used", "dulled"]
    ax.set_xticks(range(3), names, color=t["muted"])
    ax.set_yticks(range(3), names, color=t["muted"])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{share[i, j]:.0%}\n{counts[i, j]:,}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if share[i, j] > 0.45 else "#0b0b0b")
    ax.set_xlabel("Predicted class", color=t["muted"])
    ax.set_ylabel("True class", color=t["muted"])
    ax.set_title('"used" is found only 49 % of the time', color=t["text"], fontsize=11.5, loc="left", pad=10)
    save(fig, "confusion", mode)

    # 3. Ablations: change against the full model, accuracy and MSE (two panels)
    names = ["Without ALS", "Without hybrid gate", "Concatenation for C2F", "Without C2F", "Without MRTE",
             "Moving-average trend", "Without refinement", "Without RR and MRTE"]
    d_acc = [-0.008, -0.019, -0.018, -0.029, -0.026, -0.026, -0.058, -0.010]
    d_mse = [0.069, 0.046, -0.048, -0.048, 0.036, 0.103, 0.074, 0.052]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, values, noise, label, worse_positive in (
        (axes[0], d_acc, 0.012, "Δ accuracy (higher is better)", False),
        (axes[1], d_mse, 0.05, "Δ MSE (lower is better)", True),
    ):
        style(ax, t, grid_axis="x")
        ax.axvspan(-noise, noise, color=t["band"], zorder=0)
        colors = [t["neutral"] if abs(v) < noise else (orange if (v > 0) == worse_positive else aqua) for v in values]
        ax.barh(y, values, color=colors, height=0.62)
        ax.axvline(0, color=t["muted"], linewidth=0.8)
        for yi, v in zip(y, values):
            ax.text(v + (0.003 if v >= 0 else -0.003), yi, f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=t["text"])
        ax.set_xlabel(label)
    axes[0].set_yticks(y, names, color=t["muted"])
    axes[0].set_xlim(-0.075, 0.02)
    axes[1].set_xlim(-0.095, 0.13)
    axes[1].tick_params(axis="y", length=0)
    fig.suptitle("Ablations against the full model (0.676 accuracy, 0.812 MSE). Grey band: run-to-run noise",
                 color=t["text"], fontsize=11.5, x=0.01, ha="left", y=1.02)
    fig.text(0.01, -0.04, "Orange: worse than the full model   Grey: within noise (2 SD of 7 identical runs)",
             color=t["muted"], fontsize=9)
    save(fig, "ablations", mode)

    # 4. Missing-modality robustness
    groups = ["Both inputs", "Force masked\n(photo only)", "Photo masked\n(force only)"]
    none, drop = [0.659, 0.649, 0.415], [0.677, 0.641, 0.608]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    style(ax, t)
    x = np.arange(3)
    ax.bar(x - 0.19, none, 0.36, color=t["neutral"], label="Trained without dropout")
    ax.bar(x + 0.19, drop, 0.36, color=blue, label="Trained with modality dropout 0.2")
    for xi, a, b in zip(x, none, drop):
        ax.text(xi - 0.19, a + 0.008, f"{a:.3f}", ha="center", fontsize=9, color=t["text"])
        ax.text(xi + 0.19, b + 0.008, f"{b:.3f}", ha="center", fontsize=9, color=t["text"])
    ax.set_xticks(x, groups, color=t["muted"])
    ax.set_ylim(0.3, 0.79)
    ax.set_ylabel("Accuracy")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    for text in leg.get_texts():
        text.set_color(t["text"])
    ax.set_title("Dropout teaches the force stream to classify on its own", color=t["text"], fontsize=11.5,
                 loc="left", pad=10)
    save(fig, "robustness", mode)

    # 5. Run-to-run noise: seven identical runs
    runs = [0.6732, 0.6746, 0.6754, 0.6760, 0.6587, 0.6678, 0.6681]
    fig, ax = plt.subplots(figsize=(7.6, 2.4))
    style(ax, t, grid_axis="x")
    ax.scatter(runs, np.zeros(len(runs)), s=90, color=blue, zorder=3, edgecolor="none",
               label="7 runs, identical settings and seed")
    ax.scatter([0.6761], [0.25], s=110, marker="D", color=orange, zorder=3, label="Their average (ensemble)")
    ax.scatter([0.677], [-0.25], s=90, marker="s", color=t["neutral"], zorder=3,
               label="Modality dropout run (+0.018 vs its pair)")
    ax.annotate("", xy=(0.6587, -0.12), xytext=(0.6760, -0.12),
                arrowprops={"arrowstyle": "<->", "color": t["muted"], "lw": 1})
    ax.text(0.6673, -0.2, "0.017 apart", ha="center", va="top", fontsize=9, color=t["muted"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlim(0.654, 0.684)
    ax.set_xlabel("Test accuracy (mean of 10 folds)")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for text in leg.get_texts():
        text.set_color(t["text"])
    ax.set_title("Single runs are noisy: the dropout 'gain' sits inside this spread", color=t["text"],
                 fontsize=11.5, loc="left", pad=10)
    save(fig, "replicates", mode)

    # 6. Photo probe: resolution vs crop
    settings = ["Whole photo 128 × 384 (main runs)", "Whole photo 256 × 768", "Whole photo 512 × 1536",
                "Edge crop 128 × 640", "Edge crop 256 × 1280"]
    accuracy = [0.636, 0.642, 0.608, 0.652, 0.635]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    style(ax, t, grid_axis="x")
    y = np.arange(len(settings))[::-1]
    colors = [blue if a == max(accuracy) else t["neutral"] for a in accuracy]
    ax.barh(y, accuracy, color=colors, height=0.6)
    for yi, a in zip(y, accuracy):
        ax.text(a + 0.002, yi, f"{a:.3f}", va="center", fontsize=9, color=t["text"])
    ax.set_yticks(y, settings, color=t["muted"])
    ax.set_xlim(0.58, 0.67)
    ax.set_xlabel("Leave-one-tool-out accuracy of a linear probe on frozen EfficientNet-B0 features")
    ax.set_title("More pixels did not help the frozen encoder; cropping to the cutting edge did",
                 color=t["text"], fontsize=11.5, loc="left", pad=10)
    save(fig, "photo_probe", mode)

    # 7. Decision rule: argmax vs ordinal + monotone (main run)
    metrics = ['"used" recall', "Macro precision", "Accuracy", "Windows whose\nclass flickers"]
    before, after = [0.481, 0.696, 0.673, 0.40], [0.712, 0.732, 0.672, 0.12]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style(ax, t)
    x = np.arange(4)
    ax.bar(x - 0.19, before, 0.36, color=t["neutral"], label="argmax (paper's rule)")
    ax.bar(x + 0.19, after, 0.36, color=blue, label="ordinal + monotone rule")
    for xi, a, b, m in zip(x, before, after, metrics):
        fmt = "{:.0%}" if "flickers" in m else "{:.3f}"
        ax.text(xi - 0.19, a + 0.012, fmt.format(a), ha="center", fontsize=9, color=t["text"])
        ax.text(xi + 0.19, b + 0.012, fmt.format(b), ha="center", fontsize=9, color=t["text"])
    ax.set_xticks(x, metrics, color=t["muted"])
    ax.set_ylim(0, 0.85)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    for text in leg.get_texts():
        text.set_color(t["text"])
    ax.set_title("Same predictions, better decisions: respecting the order of wear", color=t["text"],
                 fontsize=11.5, loc="left", pad=10)
    save(fig, "decision_rule", mode)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
