"""Create auditable Q3 ensemble figures and a validation candidate audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
Q2 = ROOT / "03_Results" / "e" / "question-two"
Q3 = ROOT / "03_Results" / "e" / "question-three"
EXPL = Q3 / "q3-balanced-ensemble-explanations-v1.1"
OUT = Q3 / "q3-balanced-ensemble-figures-v1.1"
STATS = Q2 / "2026-09-24_q2-normalization_v1.0.npz"
SEEDS = (20260924, 20260925, 20260926)
MODALITIES = ("text", "audio", "vision")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def stable_prob(logits):
    q = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(q)
    return p / p.sum(axis=-1, keepdims=True)


def validation_audit(torch, helper, temporal, loader, robust, norm, cache, device):
    official = loader.load_official()
    valid = official["valid"]
    scaler = norm.MultimodalStandardizer.load(STATS)
    valid_text = np.load(cache.OUT / "valid.npy", mmap_mode="r", allow_pickle=False)
    arrays, observed = robust.normalize_once(
        valid, {"text": valid_text, "audio": valid.audio, "vision": valid.vision}, scaler
    )
    features, content, masks = helper.tensors(arrays, valid.content_mask, observed, torch, device)
    jobs = helper.checkpoint_jobs(torch, temporal)
    for job in jobs:
        job["model"].to(device)
    order = sorted(range(len(valid)), key=lambda i: hashlib.sha256(str(valid.ids[i]).encode()).digest())
    selected = order[:100]
    rows = []
    for count, row in enumerate(selected, 1):
        onef = {m: features[m][row:row + 1] for m in MODALITIES}
        onec = content[row:row + 1]
        oneo = {m: masks[m][row:row + 1] for m in MODALITIES}
        base_logits, _, base_prob, evidence_rows = helper.ens_predict(jobs, onef, onec, oneo, torch)
        target = int(base_prob[0].argmax())
        indices = np.flatnonzero(oneo["text"][0].detach().cpu().numpy())
        scores = {
            int(pos): float(np.mean([
                r[2]["modality_gate"][0, int(pos), 0] * r[2]["time_weight"][0, int(pos)]
                for r in evidence_rows
            ]))
            for pos in indices
        }
        if not len(indices):
            continue
        candidate = min(indices, key=lambda p: (-scores[int(p)], int(p)))
        digest = hashlib.sha256(str(valid.ids[row]).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        random_pos = int(rng.choice(indices))

        def drop(pos):
            drops = []
            for job in jobs:
                changed = {m: mask.clone() for m, mask in oneo.items()}
                changed["text"][:, int(pos)] = False
                logits, _, _ = helper.predict(job, onef, onec, changed, torch)
                p = stable_prob(logits)
                drops.append(float(base_prob[0, target] - p[0, target]))
            return float(np.mean(drops))

        candidate_delta = drop(candidate)
        random_delta = drop(random_pos)
        rows.append({
            "sample_id": str(valid.ids[row]), "row_0based": row, "target_class": target,
            "candidate_position_0based": int(candidate), "candidate_gate_time": scores[int(candidate)],
            "candidate_signed_probability_delta": candidate_delta,
            "random_position_0based": random_pos,
            "random_signed_probability_delta": random_delta,
            "signed_delta_gap": candidate_delta - random_delta,
            "candidate_above_random": candidate_delta > random_delta,
        })
        if count % 10 == 0:
            print(f"Validation audit: {count}/100", flush=True)
    return rows


def configure_plot():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10,
                         "figure.dpi": 140, "savefig.dpi": 220})
    return plt


def save_both(fig, stem):
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")


def plot_modality_influence(plt, explanation_rows):
    import matplotlib.pyplot as _plt
    values = {m: [] for m in MODALITIES}
    for row in explanation_rows:
        effects = json.loads(row["modality_effects_json"])
        for m in MODALITIES:
            values[m].append(float(effects[m]["relative_influence"]))
    means = [float(np.mean(values[m])) for m in MODALITIES]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})
    axes[0].bar(MODALITIES, means, color=["#2f6f9f", "#d27c2c", "#4b8b5b"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Mean normalized absolute effect")
    axes[0].set_title("Across 20 Attachment-4 samples")
    axes[0].grid(axis="y", alpha=.25)
    x = np.arange(len(MODALITIES))
    for j, m in enumerate(MODALITIES):
        axes[1].scatter(np.full(len(values[m]), j), values[m], s=24, alpha=.72,
                        color=["#2f6f9f", "#d27c2c", "#4b8b5b"][j], label=m)
        axes[1].plot([j - .15, j + .15], [means[j], means[j]], color="black", lw=2)
    axes[1].set_xticks(x, MODALITIES)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Per-sample normalized absolute effect")
    axes[1].set_title("Per-sample spread")
    axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Q3: leave-one-modality-out sensitivity of frozen ensemble", y=1.02)
    fig.text(.5, -.02, "Absolute probability change is normalized within each sample; this is model sensitivity, not causal proof.", ha="center", fontsize=9)
    fig.tight_layout()
    save_both(fig, "q3_modality_influence_20")
    _plt.close(fig)


def plot_candidate_audit(plt, rows):
    cand = np.asarray([float(r["candidate_signed_probability_delta"]) for r in rows])
    rnd = np.asarray([float(r["random_signed_probability_delta"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, (a, b) in enumerate(zip(cand, rnd)):
        axes[0].plot([0, 1], [a, b], color="#999999", alpha=.35, lw=.8)
    axes[0].scatter(np.zeros_like(cand), cand, s=18, label="gate/time candidate", color="#2f6f9f")
    axes[0].scatter(np.ones_like(rnd), rnd, s=18, label="deterministic random", color="#d27c2c")
    axes[0].axhline(0, color="black", lw=.8)
    axes[0].set_xticks([0, 1], ["candidate", "random"])
    axes[0].set_ylabel("Signed change in target probability")
    axes[0].set_title("Per-sample signed occlusion effects")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=.25)
    axes[1].scatter(rnd, cand, s=24, alpha=.75, color="#4b8b5b")
    lo, hi = float(min(cand.min(), rnd.min())), float(max(cand.max(), rnd.max()))
    axes[1].plot([lo, hi], [lo, hi], "--", color="#555555", lw=1)
    axes[1].set_xlabel("Random signed delta")
    axes[1].set_ylabel("Candidate signed delta")
    axes[1].set_title(f"Candidate > random: {np.mean(cand > rnd):.1%}")
    axes[1].grid(alpha=.25)
    fig.suptitle("Q3 validation audit: gate/time is a candidate ranking", y=1.02)
    fig.text(.5, -.02, "Removing one frozen-BERT feature position measures predictor sensitivity; it does not establish human or causal importance.", ha="center", fontsize=9)
    fig.tight_layout()
    save_both(fig, "q3_gate_candidate_vs_random_valid")
    plt.close(fig)


def plot_example04(plt, explanation_rows, position_rows):
    item = next(r for r in explanation_rows if r["file_id"] == "04")
    effects = json.loads(item["modality_effects_json"])
    primary = item["primary_modality"]
    local = [r for r in position_rows if r["file_id"] == "04" and r["modality"] == primary]
    local.sort(key=lambda r: int(r["position_0based"]))
    labels = [r["text_fragment"] or f"p{r['position_0based']}" for r in local]
    deltas = [float(r["class_probability_delta"]) for r in local]
    colors = ["#c94c4c" if d < 0 else "#4b8b5b" for d in deltas]
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], width_ratios=[1.1, 1.9])
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    text = (f"Attachment 4 / file 04\n\n"
            f"Prediction: {item['polarity']}\n"
            f"Intensity: {float(item['intensity']):.3f}\n"
            f"Target-class probability: {float(item['predicted_class_probability']):.3f}\n"
            f"Primary sensitivity modality: {primary}\n\n"
            f"Text: {item['raw_text']}")
    ax.text(0, 1, text, va="top", ha="left", wrap=True, fontsize=10,
            bbox={"boxstyle": "round,pad=.6", "facecolor": "#f5f5f5", "edgecolor": "#bbbbbb"})
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(MODALITIES, [effects[m]["class_probability_delta"] for m in MODALITIES],
            color=["#2f6f9f", "#d27c2c", "#4b8b5b"])
    ax2.axhline(0, color="black", lw=.8)
    ax2.set_ylabel("Signed target-probability delta")
    ax2.set_title("Leave-one-modality-out effect")
    ax2.grid(axis="y", alpha=.25)
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(labels))
    ax3.bar(x, deltas, color=colors)
    ax3.axhline(0, color="black", lw=.8)
    ax3.set_xticks(x, labels, rotation=55, ha="right")
    ax3.set_ylabel("Signed target-probability delta")
    ax3.set_xlabel(f"Observed positions in primary modality ({primary}); token labels are text offsets where available")
    ax3.set_title("Local position occlusion effects")
    ax3.grid(axis="y", alpha=.25)
    fig.suptitle("Q3 example explanation card: file 04", y=.995)
    fig.tight_layout()
    save_both(fig, "q3_example04_explanation_card")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    if not EXPL.is_dir():
        raise FileNotFoundError(EXPL)
    explanation_rows = read_csv(EXPL / "attachment4_explanations.csv")
    position_rows = read_csv(EXPL / "attachment4_position_effects.csv")
    if len(explanation_rows) != 20 or not position_rows:
        raise RuntimeError("Unexpected Q3 explanation row counts")
    if args.check:
        print(f"Preflight OK: {len(explanation_rows)} explanations, {len(position_rows)} position effects")
        print("No validation inference or figure writing")
        return
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    import torch
    temporal = load_module("q3_fig_temporal", SRC / "2026-09-24_train-q2-temporal_v1.0.py")
    loader = load_module("q3_fig_loader", SRC / "2026-09-24_q2_data_v1.0.py")
    robust = load_module("q3_fig_robust", SRC / "2026-09-24_train-q2-robust_v1.0.py")
    norm = load_module("q3_fig_norm", SRC / "2026-09-24_q2-normalization_v1.0.py")
    cache = load_module("q3_fig_cache", SRC / "2026-09-24_prepare-q2-text_v1.0.py")
    helper = load_module("q3_fig_helper", SRC / "2026-09-25_eval-q3-balanced-ensemble-explanations_v1.0.py")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    OUT.mkdir(parents=True)
    audit_rows = validation_audit(torch, helper, temporal, loader, robust, norm, cache, args.device)
    write_csv(OUT / "valid_gate_candidate_audit.csv", audit_rows, list(audit_rows[0]))
    plt = configure_plot()
    plot_modality_influence(plt, explanation_rows)
    plot_candidate_audit(plt, audit_rows)
    plot_example04(plt, explanation_rows, position_rows)
    summary = {
        "schema": "q3_balanced_ensemble_figures_v1.1",
        "source_explanations_sha256": sha(EXPL / "attachment4_explanations.csv"),
        "source_position_effects_sha256": sha(EXPL / "attachment4_position_effects.csv"),
        "validation_audit_n": len(audit_rows),
        "candidate_rule": "highest mean modality_gate times time_weight among observed text positions",
        "random_rule": "deterministic id-hash sampled observed text position",
        "candidate_mean_signed_delta": float(np.mean([float(r["candidate_signed_probability_delta"]) for r in audit_rows])),
        "random_mean_signed_delta": float(np.mean([float(r["random_signed_probability_delta"]) for r in audit_rows])),
        "candidate_above_random_fraction": float(np.mean([r["candidate_above_random"] == "True" for r in audit_rows])),
        "causal_claim": "none; occlusion is predictor sensitivity and gate/time is a candidate ranking",
        "time_mapping_status": "aligned position only; seconds unverified",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Q3 figures and validation audit saved: {OUT}")


if __name__ == "__main__":
    main()
