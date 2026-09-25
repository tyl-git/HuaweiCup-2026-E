"""Plot audited Q2 model and text-gap comparisons for the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
Q2 = ROOT / "03_Results" / "e" / "question-two"
OUT = ROOT / "03_Results" / "e" / "paper-assets-v1.0"
CONDITIONS = ("20pct_start", "20pct_middle", "20pct_end",
              "40pct_start", "40pct_middle", "40pct_end")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = OUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8")
    plt.close(fig)


def model_comparison():
    original = read(Q2 / "q2-temporal-final-v1.0" / "summary.json")
    balanced = read(Q2 / "q2-temporal-balanced-sqrt-ensemble-final-v1.0" / "summary.json")
    text_safe = read(Q2 / "q2-text-safe-final-v1.0" / "summary.json")
    original_rows = [row for row in original["results"] if row["model"] == "temporal"]
    if len(original_rows) != 3:
        raise RuntimeError("Original temporal test results are incomplete")
    original_f1 = np.mean([row["report"]["classification"]["macro_f1"]
                           for row in original_rows])
    original_mae = np.mean([row["report"]["regression"]["mae"]
                            for row in original_rows])
    f1 = [original_f1, balanced["official_test"]["classification"]["macro_f1"],
          text_safe["ensemble_report"]["classification"]["macro_f1"]]
    mae = [original_mae, balanced["official_test"]["regression"]["mae"],
           text_safe["ensemble_report"]["regression"]["mae"]]
    labels = ["Original temporal\n3-seed mean", "Balanced temporal\n3-seed ensemble",
              "Input-level gap training\n3-seed ensemble"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    colors = ("#687b83", "#167d72", "#d4913c")
    for ax, values, title, ylim, direction in (
        (axes[0], f1, "Official test Macro-F1", (0.55, 0.68), "Higher is better"),
        (axes[1], mae, "Official test MAE", (0.59, 0.65), "Lower is better"),
    ):
        bars = ax.bar(x, values, color=colors, width=0.6)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=12)
        ax.text(0.98, 0.96, direction, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#4c5a5e")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e9e8", linewidth=0.8)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Q2 complete-input performance", fontsize=13)
    save(fig, "q2_balanced_and_text_safe_test_v1.0")


def text_gap_comparison():
    report = read(Q2 / "q2-text-safe-eval-balanced-sqrt-v1.0" / "report.json")
    rows = report["rows"]
    post, input_level = [], []
    for condition in CONDITIONS:
        for encoding, target in (("post_bert", post), ("input_level", input_level)):
            values = [row["macro_f1"] for row in rows
                      if row["condition"] == condition and row["encoding"] == encoding]
            if len(values) != 3:
                raise RuntimeError(f"Missing {condition}/{encoding} rows")
            target.append(float(np.mean(values)))
    x = np.arange(len(CONDITIONS))
    fig, ax = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    ax.plot(x, post, marker="o", linewidth=2, color="#687b83", label="Post-BERT row masking")
    ax.plot(x, input_level, marker="s", linewidth=2, color="#167d72", label="[MASK] before frozen BERT")
    ax.set_xticks(x, [item.replace("pct_", "%\n") for item in CONDITIONS], fontsize=9)
    ax.set_ylabel("Validation Macro-F1 (3-seed mean)")
    ax.set_ylim(0.43, 0.63)
    ax.set_title("Identical text intervals, different masking stage", fontsize=13)
    ax.grid(axis="y", color="#e5e9e8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "q2_text_gap_protocol_comparison_v1.0")


if __name__ == "__main__":
    model_comparison()
    text_gap_comparison()
    print(f"Saved Q2 improvement figures to {OUT}")
