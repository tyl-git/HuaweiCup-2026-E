"""Build fair, saved-output-only comparisons; never train or select a model.

The official test has already been inspected during development. These are
descriptive comparisons, not an untouched confirmatory evaluation. Class-loss
comparisons use three individual-seed metrics on BOTH sides. Validation-only
cluster bootstrap resamples source video IDs and retains every clip in each
sampled video; its interval is conditional on these fixed trained models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
Q2 = ROOT / "03_Results/e/question-two"
OUT = Q2 / "q2-improvement-evidence-v1.1"
SEEDS = (20260924, 20260925, 20260926)
METRICS = ("accuracy", "macro_f1", "neutral_f1", "mae", "pearson_r")
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260925


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def metrics(y, logits, strength, pred):
    c = np.bincount(3 * y.astype(np.int64) + logits.argmax(1), minlength=9).reshape(3, 3)
    denom = c.sum(0) + c.sum(1)
    f1 = np.divide(2 * c.diagonal(), denom, out=np.zeros(3), where=denom > 0)
    a, b = strength.astype(float), pred.astype(float)
    ca, cb = a - a.mean(), b - b.mean()
    rden = np.sqrt(np.dot(ca, ca) * np.dot(cb, cb))
    return np.array([np.trace(c) / len(y), f1.mean(), f1[1],
                     np.abs(a - b).mean(), np.dot(ca, cb) / rden if rden else np.nan])


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_figure(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")


def build():
    if OUT.exists():
        raise FileExistsError(f"Output exists; run --check instead: {OUT}")
    files = {}
    loaded = {}
    rows = []
    for model, folder, summary in (
        ("temporal", "q2-temporal-final-v1.0", "summary_temporal.json"),
        ("balanced", "q2-temporal-balanced-sqrt-final-v1.0", "summary_temporal.json"),
        ("text_safe", "q2-text-safe-final-v1.0", "summary.json"),
    ):
        for split in ("valid", "test"):
            for seed in SEEDS:
                # The input-level text-safe run stores valid metrics in its
                # per-seed summary, but intentionally did not retain valid
                # logits. Keep that comparison descriptive and do not invent
                # arrays for the cluster bootstrap below.
                if model == "text_safe" and split == "valid":
                    summary_path = Q2 / "q2-text-safe-balanced-sqrt-v1.0" / f"seed_{seed}" / "metrics.json"
                    meta = read(summary_path)
                    v = meta["valid_complete"]
                    rows.append(dict(model=model, split=split, seed=seed,
                                     accuracy=float(v["classification"]["accuracy"]),
                                     macro_f1=float(v["classification"]["macro_f1"]),
                                     neutral_f1=float(v["classification"]["per_class"][1]["f1"]),
                                     mae=float(v["regression"]["mae"]),
                                     pearson_r=float(v["regression"]["pearson_r"])))
                    files[str(summary_path.relative_to(ROOT)).replace("\\\\", "/")] = sha(summary_path)
                    continue
                if split == "valid":
                    base = {"temporal": "q2-temporal-v1.0/temporal",
                            "balanced": "q2-temporal-balanced-sqrt-v1.0/temporal",
                            "text_safe": "q2-text-safe-balanced-sqrt-v1.0"}[model]
                    path = Q2 / base / f"seed_{seed}/valid_complete_predictions.npz"
                else:
                    prefix = {"temporal": "temporal", "balanced": "temporal_balanced_sqrt",
                              "text_safe": "text_safe"}[model]
                    path = Q2 / folder / f"{prefix}_seed_{seed}_test_predictions.npz"
                if not path.exists() and model == "text_safe" and split == "test":
                    matches = list((Q2 / folder).glob(f"*{seed}*predictions.npz"))
                    if len(matches) != 1:
                        raise RuntimeError(f"Need exactly one source: {matches}")
                    path = matches[0]
                with np.load(path, allow_pickle=False) as z:
                    d = {k: z[k].copy() for k in z.files}
                if len(np.unique(d["ids"])) != len(d["ids"]):
                    raise ValueError("Duplicate IDs")
                ref = loaded.get(("temporal", split, SEEDS[0]), d)
                for k in ("ids", "true_class", "true_intensity"):
                    np.testing.assert_array_equal(ref[k], d[k])
                loaded[model, split, seed] = d
                files[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
                value = metrics(d["true_class"], d["logits"], d["true_intensity"], d["intensity"])
                rows.append(dict(model=model, split=split, seed=seed,
                                 **dict(zip(METRICS, map(float, value)))))
    # Every resample is common to both models and all seeds; no model refits.
    ids = loaded["temporal", "valid", SEEDS[0]]["ids"].astype(str)
    videos = np.array([x.rsplit("$_$", 1)[0] for x in ids])
    groups = [np.flatnonzero(videos == v) for v in sorted(set(videos))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diff = []
    for _ in range(BOOTSTRAP_N):
        ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        values = {}
        for model in ("temporal", "balanced"):
            values[model] = np.mean([
                metrics(d["true_class"][ix], d["logits"][ix],
                        d["true_intensity"][ix], d["intensity"][ix])
                for seed in SEEDS for d in [loaded[model, "valid", seed]]], axis=0)
        diff.append(values["balanced"] - values["temporal"])
    difference = np.asarray(diff)
    aggregates = []
    for model in ("temporal", "balanced", "text_safe"):
        for split in ("valid", "test"):
            a = np.array([[row[k] for k in METRICS] for row in rows
                          if row["model"] == model and row["split"] == split])
            aggregates.append(dict(model=model, split=split, aggregation="mean_of_individual_seed_metrics",
                                   **{f"{k}_mean": float(a[:, j].mean()) for j, k in enumerate(METRICS)},
                                   **{f"{k}_sample_sd": float(a[:, j].std(ddof=1)) for j, k in enumerate(METRICS)}))
    confidence = {k: {"percentile_95_interval": np.nanquantile(difference[:, j], [0.025, 0.975]).tolist(),
                      "finite_resamples": int(np.isfinite(difference[:, j]).sum())}
                  for j, k in enumerate(METRICS)}
    OUT.mkdir(parents=True)
    csv_write(OUT / "seed_metrics.csv", rows)
    csv_write(OUT / "matched_seed_summary.csv", aggregates)
    np.savez_compressed(OUT / "valid_cluster_bootstrap.npz", metric_names=METRICS,
                        balanced_minus_original=difference)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#687b83", "#167d72", "#bc7427")
    labels = ("Original temporal", "Class balanced", "Input-level gap")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.3), constrained_layout=True)
    for rowno, split in enumerate(("valid", "test")):
        for col, name in enumerate(("macro_f1", "mae")):
            ax = axes[rowno, col]
            for x, model in enumerate(("temporal", "balanced", "text_safe")):
                a = np.array([r[name] for r in rows if r["model"] == model and r["split"] == split])
                ax.scatter(x + np.array([-0.08, 0, 0.08]), a, color=colors[x], alpha=.65, s=24)
                ax.errorbar(x, a.mean(), yerr=a.std(ddof=1), fmt="D", color=colors[x], capsize=5)
            ax.set_xticks(range(3), labels, rotation=12, ha="right", fontsize=9)
            ax.set_title(f"{'Validation' if split == 'valid' else 'Previously viewed official test'}: {name}", fontsize=10)
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Matched comparison: three individual seeds per method\nDots = seeds; diamond/error bar = mean +/- sample SD", fontsize=12)
    save_figure(fig, "q2_matched_comparison")
    plt.close(fig)
    main = read(Q2 / "q2-temporal-balanced-sqrt-ensemble-final-v1.1/summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), constrained_layout=True)
    for ax, key, label in zip(axes, ("validation", "official_test"), ("Validation (728)", "Previously viewed official test (727)")):
        c = np.array(main[key]["classification"]["confusion_matrix_true_rows_predicted_columns"])
        ax.imshow(c / c.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for y in range(3):
            for x in range(3):
                ax.text(x, y, f"{c[y,x]}\n{c[y,x]/c[y].sum():.1%}", ha="center", va="center",
                        color="white" if c[y,x]/c[y].sum() > .6 else "#17323d")
        ax.set_xticks(range(3), ("Negative", "Neutral", "Positive"))
        ax.set_yticks(range(3), ("Negative", "Neutral", "Positive"))
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(label, fontsize=10)
    fig.suptitle("Frozen class-balanced three-seed logit ensemble", fontsize=12)
    save_figure(fig, "q2_current_confusion")
    plt.close(fig)
    condition_names = [f"{m}_missing_{f}pct_{p}" for m in ("text", "audio", "vision")
                       for f in (20, 40) for p in ("start", "middle", "end")]
    delta = np.array([main["validation_missing"][k]["classification"]["macro_f1"]
                      - main["validation"]["classification"]["macro_f1"] for k in condition_names]).reshape(3, 6)
    fig, ax = plt.subplots(figsize=(8.8, 3.5), constrained_layout=True)
    limit = max(.01, float(np.max(np.abs(delta))))
    im = ax.imshow(delta, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for y in range(3):
        for x in range(6):
            ax.text(x, y, f"{delta[y,x]:+.3f}", ha="center", va="center",
                    color="white" if abs(delta[y,x]) > .65*limit else "black")
    ax.set_xticks(range(6), [f"{f}% {p}" for f in (20,40) for p in ("start","middle","end")])
    ax.set_yticks(range(3), ("Text features", "Audio", "Vision"))
    ax.set_title("Current ensemble: validation Macro-F1 change under feature gaps\nText gaps are after BERT; denominator follows original observed-position protocol", fontsize=10)
    fig.colorbar(im, ax=ax, label="Missing - complete Macro-F1")
    save_figure(fig, "q2_current_single_gap_heatmap"); plt.close(fig)
    history_paths = [Q2 / "q2-temporal-balanced-sqrt-v1.0/temporal" / f"seed_{s}/history.json" for s in SEEDS]
    fig, ax = plt.subplots(figsize=(7.6, 4.1), constrained_layout=True)
    for path, seed, color in zip(history_paths, SEEDS, colors):
        h = read(path)
        ax.plot([v["epoch"] for v in h], [v["valid_complete_selection_loss"] for v in h], marker="o", label=str(seed), color=color)
        files[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Unweighted validation CE + L1")
    ax.set_title("Class-balanced training: checkpoint selection curves")
    ax.legend(frameon=False); ax.grid(alpha=.2); ax.spines[["top","right"]].set_visible(False)
    save_figure(fig, "q2_balanced_learning_curves"); plt.close(fig)
    files[str((Q2 / "q2-temporal-balanced-sqrt-ensemble-final-v1.1/summary.json").relative_to(ROOT)).replace("\\", "/")] = sha(Q2 / "q2-temporal-balanced-sqrt-ensemble-final-v1.1/summary.json")
    report = {"schema": "q2_improvement_evidence_v1.1", "aggregation": "mean_of_individual_seed_metrics",
              "official_test_previously_viewed": True, "new_training": False, "model_selection": False,
              "bootstrap": {"split": "valid", "unit": "source_video_id", "video_count": len(groups),
                            "resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                            "quantity": "paired_mean_seed_metric_balanced_minus_original",
                            "interval_scope": "conditional on fixed models and reused validation data; not confirmatory and not training uncertainty",
                            "metrics": confidence}, "sources": files,
              "artifacts": {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"saved":str(OUT),"bootstrap":report["bootstrap"]}, indent=2))


def check():
    report = read(OUT / "report.json")
    for name, digest in report["sources"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in report["artifacts"].items():
        assert sha(OUT / name) == digest, name
    with (OUT / "seed_metrics.csv").open(encoding="utf-8-sig",newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows)==18
    with (OUT / "matched_seed_summary.csv").open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            subset=[r for r in rows if (r["model"],r["split"])==(row["model"],row["split"])]
            assert len(subset)==3
            for name in METRICS:
                a=np.array([float(r[name]) for r in subset])
                assert abs(a.mean()-float(row[name+"_mean"]))<1e-12
                assert abs(a.std(ddof=1)-float(row[name+"_sample_sd"]))<1e-12
    print("Evidence audit passed: 18 seed rows, 6 matched summaries, source/artifact hashes")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run",action="store_true");g.add_argument("--check",action="store_true")
    a=p.parse_args()
    build() if a.run else check()
