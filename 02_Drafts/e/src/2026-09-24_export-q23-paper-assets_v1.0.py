"""Render paper figures from frozen Q2/Q3 reports without model inference."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
Q2 = ROOT / "03_Results/e/question-two"
Q3 = ROOT / "03_Results/e/question-three"
OUT = ROOT / "03_Results/e/paper-assets-v1.0"
MODELS = ("temporal", "no_temporal", "no_gap_signal")
COLORS = {"temporal": "#187267", "no_temporal": "#C75D38", "no_gap_signal": "#65719D"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def model_comparison(final):
    grouped = defaultdict(list)
    for row in final["results"]:
        grouped[row["model"]].append(row["report"])
    assert all(len(grouped[model]) == 3 for model in MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    labels = ("Temporal + gap", "No temporal", "No gap signal")
    x = np.arange(3)
    for axis, field, title, ylabel in (
        (axes[0], lambda r: r["classification"]["macro_f1"], "Macro-F1", "Higher is better"),
        (axes[1], lambda r: r["regression"]["mae"], "MAE", "Lower is better"),
    ):
        values = [np.asarray([field(r) for r in grouped[m]]) for m in MODELS]
        axis.bar(x, [v.mean() for v in values], color=[COLORS[m] for m in MODELS], width=.63)
        axis.errorbar(x, [v.mean() for v in values],
                      yerr=[v.std(ddof=1) for v in values], fmt="none", ecolor="#252525",
                      capsize=4, lw=1)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#E1E5E5", linewidth=.7)
        axis.set_axisbelow(True)
    fig.suptitle("Official test: three fixed seeds per model", fontsize=11)
    save(fig, "q2_test_ablation_v1.0")


def missing_heatmap():
    folder = Q2 / "q2-temporal-v1.0/temporal"
    reports = [read_json(folder / f"seed_{seed}/metrics.json")
               for seed in (20260924, 20260925, 20260926)]
    conditions = [f"{fraction}% {position}" for fraction in (20, 40)
                  for position in ("start", "middle", "end")]
    modalities = ("text", "audio", "vision")
    matrix = np.empty((3, 6))
    for i, modality in enumerate(modalities):
        for j, label in enumerate(conditions):
            fraction, position = label.split()
            key = f"{modality}_missing_{fraction[:-1]}pct_{position}"
            matrix[i, j] = np.mean([
                report["valid_missing"][key]["classification"]["macro_f1"]
                - report["valid_complete"]["classification"]["macro_f1"]
                for report in reports
            ])
    scale = max(abs(matrix.min()), abs(matrix.max()))
    fig, ax = plt.subplots(figsize=(9, 2.9), layout="constrained")
    image = ax.imshow(matrix, cmap="BrBG", vmin=-scale, vmax=scale, aspect="auto")
    ax.set_xticks(np.arange(6), conditions)
    ax.set_yticks(np.arange(3), modalities)
    ax.set_title("Validation: macro-F1 change under contiguous feature gaps")
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="Delta from complete input", shrink=.82)
    save(fig, "q2_missing_heatmap_v1.0")


def explanations(summary):
    records = read_csv(Q3 / "q3-explanations-v1.0/attachment4_explanations.csv")
    assert len(records) == 20
    influences = np.asarray([[json.loads(row["modality_effects_json"])[modality]
                              ["relative_influence"] for modality in ("text", "audio", "vision")]
                             for row in records])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.subplots_adjust(top=.84)
    bottom = np.zeros(20)
    for j, (name, color) in enumerate((("Text", "#187267"), ("Audio", "#C75D38"),
                                        ("Vision", "#65719D"))):
        ax.bar(np.arange(20), influences[:, j], bottom=bottom, color=color, label=name, width=.78)
        bottom += influences[:, j]
    ax.set_xticks(np.arange(20), [row["file_id"] for row in records], fontsize=8)
    ax.set_xlabel("Attachment 4 sample ID")
    ax.set_ylabel("Relative absolute leave-one-modality-out effect")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.02))
    save(fig, "q3_modality_influence_v1.0")

    fidelity = summary["valid_fidelity"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4), layout="constrained")
    ax.bar(["Gate x time top position", "Fixed random position"],
           [fidelity["gate_time_top1_mean_drop"],
            fidelity["deterministic_random_mean_drop"]],
           color=["#187267", "#65719D"], width=.58)
    ax.set_ylabel("Mean predicted-class probability drop")
    ax.set_title("Frozen validation subset (n=100)")
    ax.grid(axis="y", color="#E1E5E5", linewidth=.7)
    ax.set_axisbelow(True)
    save(fig, "q3_occlusion_fidelity_v1.0")


def main():
    final = read_json(Q2 / "q2-temporal-final-v1.0/summary.json")
    summary = read_json(Q3 / "q3-explanations-v1.0/summary.json")
    if final["official_test_n"] != 727 or summary["valid_fidelity"]["sample_count"] != 100:
        raise ValueError("Unexpected frozen experiment scope")
    OUT.mkdir(parents=True, exist_ok=True)
    model_comparison(final)
    missing_heatmap()
    explanations(summary)
    print(f"Saved 4 PNG/SVG figures: {OUT}")


if __name__ == "__main__":
    main()
