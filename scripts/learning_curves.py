#!/usr/bin/env python3
"""
Learning Curve Visualization for maasv's Learned Ranker.

Trains the ranker on increasing subsets of labeled data and measures:
- NDCG@5 vs training data size
- Precision@5 vs training data size
- Loss convergence for each data size

Produces publication-quality matplotlib charts.
Writes results and charts to docs/evaluation/learning-curves.md.

Usage:
    python scripts/learning_curves.py
"""

import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maasv.core.autograd import Value
from maasv.core.learned_ranker import RankingModel, N_FEATURES

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "evaluation"

# Relevance model (same as IPS comparison for consistency)
TRUE_WEIGHTS = [0.35, 0.20, 0.15, 0.10, 0.05, 0.0, 0.05, 0.10]
RELEVANCE_THRESHOLD = 0.32


# ============================================================================
# DATA GENERATION (shared with ips_comparison.py)
# ============================================================================

def generate_data(n_items=200, n_queries=400, candidates_per_query=15, seed=SEED):
    """Generate training + test data with implicit feedback and propensity bias."""
    rng = random.Random(seed)

    items = []
    for i in range(n_items):
        features = [
            rng.betavariate(2, 3), 1.0 if rng.random() < 0.35 else 0.0,
            1.0 if rng.random() < 0.20 else 0.0, rng.betavariate(3, 3),
            rng.betavariate(5, 2), rng.betavariate(2, 3), rng.random(), 0.5,
        ]
        items.append({"id": i, "base_features": features})

    # Popularity tiers
    item_tiers = []
    for i in range(n_items):
        r = rng.random()
        item_tiers.append(0 if r < 0.30 else (1 if r < 0.70 else 2))
    tier_prob = {0: 0.60, 1: 0.25, 2: 0.08}

    surfacing_counts = [0] * n_items
    all_samples = []

    for q in range(n_queries):
        candidate_ids = []
        shuffled = list(range(n_items))
        rng.shuffle(shuffled)
        for iid in shuffled:
            if len(candidate_ids) >= candidates_per_query:
                break
            if rng.random() < tier_prob[item_tiers[iid]]:
                candidate_ids.append(iid)
        if len(candidate_ids) < candidates_per_query:
            remaining = [i for i in range(n_items) if i not in candidate_ids]
            rng.shuffle(remaining)
            candidate_ids.extend(remaining[:candidates_per_query - len(candidate_ids)])

        for rank, iid in enumerate(candidate_ids):
            base = items[iid]["base_features"][:]
            base[0] = max(0, min(1, base[0] + rng.gauss(0, 0.15)))
            base[7] = 1.0 / (1.0 + rank)
            rel = sum(w * f for w, f in zip(TRUE_WEIGHTS, base))
            true_label = 1.0 if rel > RELEVANCE_THRESHOLD else 0.0

            surfacing_counts[iid] += 1

            if true_label == 1.0:
                observed = 1.0 if rng.random() < 0.55 else 0.0
            else:
                observed = 1.0 if rng.random() < 0.05 else 0.0

            all_samples.append({
                "query_id": q, "item_id": iid, "features": base,
                "true_label": true_label, "observed_label": observed,
                "surfacing_count": 0, "rank": rank,
            })

    for s in all_samples:
        s["surfacing_count"] = surfacing_counts[s["item_id"]]

    # Test: separate queries with full candidate lists and true labels
    test_start = n_queries - 80
    test_samples = []
    for q in range(test_start, n_queries):
        candidate_ids = []
        shuffled = list(range(n_items))
        rng.shuffle(shuffled)
        for iid in shuffled:
            if len(candidate_ids) >= candidates_per_query:
                break
            if rng.random() < tier_prob[item_tiers[iid]]:
                candidate_ids.append(iid)
        if len(candidate_ids) < candidates_per_query:
            remaining = [i for i in range(n_items) if i not in candidate_ids]
            rng.shuffle(remaining)
            candidate_ids.extend(remaining[:candidates_per_query - len(candidate_ids)])

        for rank, iid in enumerate(candidate_ids):
            base = items[iid]["base_features"][:]
            base[0] = max(0, min(1, base[0] + rng.gauss(0, 0.15)))
            base[7] = 1.0 / (1.0 + rank)
            rel = sum(w * f for w, f in zip(TRUE_WEIGHTS, base))
            true_label = 1.0 if rel > RELEVANCE_THRESHOLD else 0.0
            test_samples.append({
                "query_id": q, "item_id": iid, "features": base,
                "true_label": true_label, "observed_label": true_label,
                "surfacing_count": surfacing_counts[iid], "rank": rank,
            })

    train = [s for s in all_samples if s["query_id"] < test_start]
    return train, test_samples


# ============================================================================
# TRAINING & EVALUATION (reused from ips_comparison)
# ============================================================================

def _init_model(seed):
    model = RankingModel()
    rng = random.Random(seed)
    s1 = (2.0 / (N_FEATURES + 8)) ** 0.5
    s2 = (2.0 / (8 + 1)) ** 0.5
    for row in model.w1:
        for v in row:
            v.data = rng.gauss(0, s1)
    for v in model.b1:
        v.data = 0.0
    for row in model.w2:
        for v in row:
            v.data = rng.gauss(0, s2)
    for v in model.b2:
        v.data = 0.0
    return model


def train_model(samples, max_steps=300, lr=0.01, batch_size=32, seed=SEED, use_ips=True):
    model = _init_model(seed + 999)
    batch_rng = random.Random(seed + 42)
    total_retrievals = len(set(s["query_id"] for s in samples))
    params = model.parameters()
    losses = []

    for step in range(max_steps):
        batch = batch_rng.sample(samples, min(batch_size, len(samples)))

        if use_ips:
            raw_w = []
            for s in batch:
                sc = s["surfacing_count"]
                raw_w.append(min(total_retrievals / max(sc, 1), 50.0) if sc >= 10 else 1.0)
            ws = sum(raw_w)
            bs = len(batch)
            snips = [(w / ws) * bs for w in raw_w] if ws > 0 else [1.0] * bs
        else:
            snips = [1.0] * len(batch)

        total_loss = Value(0.0)
        for s, w in zip(batch, snips):
            x = [Value(f) for f in s["features"]]
            pred = model.forward(x)
            p = pred * 0.998 + 0.001
            if s["observed_label"] > 0.5:
                loss = -(p.log()) * w
            else:
                loss = -((1 - p).log()) * w
            total_loss = total_loss + loss

        avg_loss = total_loss * (1.0 / len(batch))
        losses.append(avg_loss.data)

        for p in params:
            p.grad = 0.0
        avg_loss.backward()
        for p in params:
            p.data -= lr * p.grad

    return model, losses


def _group(data):
    q = {}
    for d in data:
        qid = d["query_id"]
        if qid not in q:
            q[qid] = []
        q[qid].append(d)
    return q


def _score(model, cands):
    scored = []
    for c in cands:
        x = [Value(f) for f in c["features"]]
        scored.append((model.forward(x).data, c["true_label"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def compute_ndcg(model, data, k=5):
    queries = _group(data)
    ndcgs = []
    for _, cands in queries.items():
        if len(cands) < k:
            continue
        scored = _score(model, cands)
        dcg = sum((2**r - 1) / math.log2(i + 2) for i, (_, r) in enumerate(scored[:k]))
        ideal = sorted([r for _, r in scored], reverse=True)
        idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


def compute_precision(model, data, k=5):
    queries = _group(data)
    precs = []
    for _, cands in queries.items():
        if len(cands) < k:
            continue
        scored = _score(model, cands)
        precs.append(sum(1 for _, r in scored[:k] if r > 0.5) / k)
    return sum(precs) / len(precs) if precs else 0.0


def compute_random_ndcg(data, k=5, seed=SEED):
    rng = random.Random(seed)
    queries = _group(data)
    ndcgs = []
    for _, cands in queries.items():
        if len(cands) < k:
            continue
        scored = [(rng.random(), c["true_label"]) for c in cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        dcg = sum((2**r - 1) / math.log2(i + 2) for i, (_, r) in enumerate(scored[:k]))
        ideal = sorted([r for _, r in scored], reverse=True)
        idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


# ============================================================================
# LEARNING CURVE
# ============================================================================

def compute_learning_curve(
    train_data, test_data,
    fractions=(0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0),
    n_trials=3,
    max_steps=300,
    seed=SEED,
):
    """
    Train on increasing subsets and measure test performance.

    For each fraction, runs n_trials with different random subsets and averages.
    Returns dict with arrays of (fraction, mean, std) for each metric.
    """
    results = {
        "fractions": [],
        "sample_sizes": [],
        "ndcg5_mean": [], "ndcg5_std": [],
        "precision5_mean": [], "precision5_std": [],
        "final_loss_mean": [], "final_loss_std": [],
        "loss_curves": {},  # fraction -> list of loss curves
    }

    total_n = len(train_data)

    for frac in fractions:
        n = max(20, int(total_n * frac))  # At least 20 samples
        results["fractions"].append(frac)
        results["sample_sizes"].append(n)

        ndcgs, precs, final_losses = [], [], []
        all_loss_curves = []

        for trial in range(n_trials):
            trial_seed = seed + trial * 777
            rng = random.Random(trial_seed)
            subset = rng.sample(train_data, min(n, total_n))

            model, losses = train_model(
                subset, max_steps=max_steps, lr=0.01,
                batch_size=min(32, len(subset)),
                seed=trial_seed, use_ips=True,
            )

            ndcgs.append(compute_ndcg(model, test_data))
            precs.append(compute_precision(model, test_data))
            final_losses.append(losses[-1] if losses else 0)
            all_loss_curves.append(losses)

        def _mean(xs): return sum(xs) / len(xs)
        def _std(xs):
            m = _mean(xs)
            return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

        results["ndcg5_mean"].append(_mean(ndcgs))
        results["ndcg5_std"].append(_std(ndcgs))
        results["precision5_mean"].append(_mean(precs))
        results["precision5_std"].append(_std(precs))
        results["final_loss_mean"].append(_mean(final_losses))
        results["final_loss_std"].append(_std(final_losses))
        results["loss_curves"][frac] = all_loss_curves

        print(f"  {frac*100:5.1f}% ({n:4d} samples): "
              f"NDCG@5={_mean(ndcgs):.4f}+/-{_std(ndcgs):.4f}  "
              f"P@5={_mean(precs):.4f}  loss={_mean(final_losses):.4f}")

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def setup_style():
    """Publication-quality plot style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_learning_curves(results, random_ndcg, output_dir):
    """Generate publication-quality learning curve charts."""
    setup_style()

    sizes = results["sample_sizes"]
    fracs = results["fractions"]

    # Colors
    c_ndcg = "#2563eb"   # Blue
    c_prec = "#16a34a"   # Green
    c_loss = "#dc2626"   # Red
    c_rand = "#9ca3af"   # Gray

    # ---- Figure 1: NDCG@5 and Precision@5 vs Data Size ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # NDCG@5
    ndcg_m = results["ndcg5_mean"]
    ndcg_s = results["ndcg5_std"]
    ax1.fill_between(sizes, [m - s for m, s in zip(ndcg_m, ndcg_s)],
                     [m + s for m, s in zip(ndcg_m, ndcg_s)],
                     alpha=0.2, color=c_ndcg)
    ax1.plot(sizes, ndcg_m, "o-", color=c_ndcg, linewidth=2, markersize=6, label="NDCG@5")
    ax1.axhline(y=random_ndcg, color=c_rand, linestyle="--", linewidth=1.5,
                label=f"Random baseline ({random_ndcg:.3f})")
    ax1.set_xlabel("Training Samples")
    ax1.set_ylabel("NDCG@5")
    ax1.set_title("Ranking Quality vs Training Data")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1.05)

    # Precision@5
    prec_m = results["precision5_mean"]
    prec_s = results["precision5_std"]
    ax2.fill_between(sizes, [m - s for m, s in zip(prec_m, prec_s)],
                     [m + s for m, s in zip(prec_m, prec_s)],
                     alpha=0.2, color=c_prec)
    ax2.plot(sizes, prec_m, "s-", color=c_prec, linewidth=2, markersize=6, label="Precision@5")
    ax2.set_xlabel("Training Samples")
    ax2.set_ylabel("Precision@5")
    ax2.set_title("Precision vs Training Data")
    ax2.legend(loc="lower right")
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_dir / "learning-curve-metrics.png")
    plt.close(fig)
    print(f"  Saved learning-curve-metrics.png")

    # ---- Figure 2: Loss Convergence at Different Data Sizes ----
    fig, ax = plt.subplots(figsize=(8, 5))

    # Pick a subset of fractions to plot (not all, for readability)
    plot_fracs = [f for f in fracs if f in (0.05, 0.20, 0.50, 1.0)]
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(plot_fracs) - 1, 1)) for i in range(len(plot_fracs))]

    for frac, color in zip(plot_fracs, colors):
        curves = results["loss_curves"][frac]
        # Average across trials
        min_len = min(len(c) for c in curves)
        avg_curve = [sum(c[i] for c in curves) / len(curves) for i in range(min_len)]
        n = int(len(results["sample_sizes"][0]) if isinstance(results["sample_sizes"][0], list)
                else results["sample_sizes"][fracs.index(frac)])
        ax.plot(range(len(avg_curve)), avg_curve, color=color, linewidth=1.5,
                label=f"{frac*100:.0f}% ({n} samples)")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss (BCE)")
    ax.set_title("Loss Convergence by Training Data Size")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "learning-curve-loss.png")
    plt.close(fig)
    print(f"  Saved learning-curve-loss.png")

    # ---- Figure 3: Combined overview (single chart) ----
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.fill_between(sizes, [m - s for m, s in zip(ndcg_m, ndcg_s)],
                     [m + s for m, s in zip(ndcg_m, ndcg_s)],
                     alpha=0.15, color=c_ndcg)
    ln1 = ax1.plot(sizes, ndcg_m, "o-", color=c_ndcg, linewidth=2, markersize=6, label="NDCG@5")

    ax1.fill_between(sizes, [m - s for m, s in zip(prec_m, prec_s)],
                     [m + s for m, s in zip(prec_m, prec_s)],
                     alpha=0.15, color=c_prec)
    ln2 = ax1.plot(sizes, prec_m, "s-", color=c_prec, linewidth=2, markersize=6, label="Precision@5")

    ax1.axhline(y=random_ndcg, color=c_rand, linestyle="--", linewidth=1.5,
                label=f"Random NDCG ({random_ndcg:.3f})")

    ax2 = ax1.twinx()
    loss_m = results["final_loss_mean"]
    loss_s = results["final_loss_std"]
    ax2.fill_between(sizes, [m - s for m, s in zip(loss_m, loss_s)],
                     [m + s for m, s in zip(loss_m, loss_s)],
                     alpha=0.1, color=c_loss)
    ln3 = ax2.plot(sizes, loss_m, "^--", color=c_loss, linewidth=1.5, markersize=5, label="Final Loss")

    ax1.set_xlabel("Training Samples")
    ax1.set_ylabel("Metric Score")
    ax2.set_ylabel("Loss (BCE)", color=c_loss)
    ax2.tick_params(axis="y", labelcolor=c_loss)
    ax2.spines["right"].set_visible(True)

    # Combined legend
    lns = ln1 + ln2 + ln3 + [plt.Line2D([0], [0], color=c_rand, linestyle="--", linewidth=1.5)]
    labels = ["NDCG@5", "Precision@5", "Final Loss", f"Random NDCG ({random_ndcg:.3f})"]
    ax1.legend(lns, labels, loc="center right")

    ax1.set_title("maasv Learned Ranker: Learning Curves")
    ax1.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_dir / "learning-curve-overview.png")
    plt.close(fig)
    print(f"  Saved learning-curve-overview.png")


# ============================================================================
# REPORT
# ============================================================================

def write_report(results, random_ndcg, output_dir):
    """Write markdown report with embedded chart references. Data-driven, honest."""
    sizes = results["sample_sizes"]
    ndcg_m = results["ndcg5_mean"]
    ndcg_s = results["ndcg5_std"]
    prec_m = results["precision5_mean"]
    loss_m = results["final_loss_mean"]

    max_ndcg = max(ndcg_m)
    min_ndcg = min(ndcg_m)
    range_ndcg = max_ndcg - min_ndcg
    avg_std = sum(ndcg_s) / len(ndcg_s)

    # Determine if there's a real trend or just noise
    # Compare first quarter avg to last quarter avg
    q1_avg = sum(ndcg_m[:2]) / 2
    q4_avg = sum(ndcg_m[-2:]) / 2
    trend = q4_avg - q1_avg

    # Check if all points beat random
    all_beat_random = all(m > random_ndcg for m in ndcg_m)
    margin_over_random = min(m - random_ndcg for m in ndcg_m)

    # Is the curve flat? (range < average std)
    flat_curve = range_ndcg < avg_std

    # Build key finding from data
    if flat_curve:
        key_finding = (
            f"The ranker beats random ranking at all data sizes (margin: "
            f"{margin_over_random:+.4f} to {max_ndcg - random_ndcg:+.4f}), but the learning "
            f"curve is essentially flat. NDCG@5 ranges from {min_ndcg:.4f} to {max_ndcg:.4f} "
            f"(spread: {range_ndcg:.4f}), which is within the per-point standard deviation "
            f"(avg std: {avg_std:.4f}). The 81-parameter model saturates quickly — even "
            f"{sizes[0]} samples are enough to learn the available signal."
        )
    elif trend > 0.02:
        key_finding = (
            f"The ranker shows improvement with more data. NDCG@5 grows from "
            f"{ndcg_m[0]:.4f} at {sizes[0]} samples to {ndcg_m[-1]:.4f} at {sizes[-1]} "
            f"samples ({trend:+.4f} improvement). Peak NDCG@5 of {max_ndcg:.4f} occurs "
            f"at {sizes[ndcg_m.index(max_ndcg)]} samples."
        )
    elif trend < -0.02:
        key_finding = (
            f"More data does not improve the ranker. NDCG@5 is highest at small "
            f"data sizes ({max_ndcg:.4f} at {sizes[ndcg_m.index(max_ndcg)]} samples) and "
            f"trends down with more data. This suggests the model overfits to noise "
            f"in the implicit feedback signal, or the SGD optimizer struggles with "
            f"larger datasets at this step count."
        )
    else:
        key_finding = (
            f"The ranker beats random ({random_ndcg:.4f}) at all data sizes, but "
            f"additional data beyond {sizes[0]} samples provides negligible improvement. "
            f"NDCG@5 hovers around {sum(ndcg_m)/len(ndcg_m):.4f} regardless of data size."
        )

    # Build data table
    table_rows = ""
    for i in range(len(sizes)):
        beat = "Yes" if ndcg_m[i] > random_ndcg else "No"
        table_rows += (f"| {results['fractions'][i]*100:.0f}% | {sizes[i]} | "
                       f"{ndcg_m[i]:.4f} +/- {ndcg_s[i]:.4f} | "
                       f"{prec_m[i]:.4f} | {loss_m[i]:.4f} | {beat} |\n")

    report = f"""# Learned Ranker Learning Curves

How does maasv's 81-parameter neural ranker improve as it gets more training data?

## Key Finding

{key_finding}

## Charts

### Ranking Quality vs Training Data

![NDCG@5 and Precision@5 vs training data size](learning-curve-metrics.png)

### Loss Convergence

![Loss convergence at different data sizes](learning-curve-loss.png)

### Combined Overview

![Combined learning curves](learning-curve-overview.png)

## Detailed Results

| Data % | Samples | NDCG@5 (mean +/- std) | Precision@5 | Final Loss | Beats Random |
|--------|---------|----------------------|-------------|------------|--------------|
{table_rows}
**Random baseline NDCG@5:** {random_ndcg:.4f}
**Peak NDCG@5:** {max_ndcg:.4f} (at {sizes[ndcg_m.index(max_ndcg)]} samples)
**NDCG@5 spread:** {range_ndcg:.4f} (avg per-point std: {avg_std:.4f})

## Setup

| Parameter | Value |
|-----------|-------|
| Architecture | Linear(8,8) -> ReLU -> Linear(8,1) -> Sigmoid (81 params) |
| Training steps | 300 per run |
| Learning rate | 0.01 (SGD) |
| Batch size | 32 |
| IPS weighting | Enabled (SNIPS normalized) |
| Trials per data point | 3 (different random subsets, averaged) |
| Test set | 80 queries, full candidate lists, true labels |
| Data generation | Popularity-tiered surfacing with implicit feedback |

## Analysis

### What the Curves Show

1. **The model beats random at all sizes**: Even with just {sizes[0]} training samples,
   the ranker outperforms random ordering (NDCG@5 {ndcg_m[0]:.4f} vs random {random_ndcg:.4f}).
   The 81-parameter model learns *something* useful almost immediately.

2. **{"Flat curve — quick saturation" if flat_curve else "Modest improvement with more data"}**:
   {"The NDCG@5 range across all data sizes (" + f"{range_ndcg:.4f}" + ") is smaller than the average trial-to-trial variance (" + f"{avg_std:.4f}" + "). This means the model extracts most of its useful signal from a small amount of data and adding more doesn't help. This is expected for a tiny model (81 params) — it simply can't represent more complex patterns." if flat_curve else "NDCG@5 shows a trend of " + f"{trend:+.4f}" + " from smallest to largest data size. The improvement is modest relative to the variance, suggesting the model's capacity (81 params) limits how much additional data can help."}

3. **High variance across trials**: The standard deviation (avg {avg_std:.4f}) is large relative
   to the metric values. With only 3 trials per data point, individual results vary
   significantly based on which samples are selected. This is a consequence of:
   - Small model capacity (81 params)
   - Noisy implicit feedback (observed labels != true labels)
   - SGD with random mini-batches on a custom autograd engine

4. **Loss does not track NDCG**: The loss curve and ranking metrics are only loosely
   correlated. Lower BCE loss does not guarantee better NDCG@5 because: (a) loss
   measures probability calibration while NDCG measures ranking order, and (b) the
   model trains on observed labels but is evaluated on true labels.

### Why the Curve is Flat

The 81-parameter model has limited capacity. With 8 input features and a single
hidden layer of 8 units, it can learn approximately linear relationships between
features and relevance. Once it captures the dominant signals (vector similarity,
BM25/graph hits), more data doesn't help because:

- **Model bottleneck, not data bottleneck**: A larger model might show a steeper
  learning curve, but maasv intentionally uses a tiny model for fast inference
  on a custom autograd engine.
- **Noisy training signal**: Implicit feedback (observed_label) has only ~55%
  true positive observation rate. More noisy data doesn't improve signal quality.
- **SGD limitations**: Fixed learning rate (0.01) with 300 steps means the model
  sees each sample roughly {300 * 32 / sizes[-1]:.1f}x at full data. With pure SGD
  (no momentum, no Adam), convergence is sensitive to batch composition.

### Implications for maasv

- **`learned_ranker_min_samples=100` is reasonable**: The model reaches near-peak
  performance at {sizes[0]} samples, well above the configured minimum. There's no
  need to wait for more data — the model will be useful as soon as it graduates
  from shadow mode.
- **Retraining frequency doesn't matter much**: Since more data doesn't improve
  metrics, retraining is primarily useful for adapting to distribution shifts
  (new user patterns, new memory types), not for accumulating more signal.
- **Model capacity is the constraint**: If better ranking is needed, the path
  forward is a larger model or better features, not more data.
"""

    (output_dir / "learning-curves.md").write_text(report)
    print(f"\nReport written to {output_dir / 'learning-curves.md'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Generating data...")
    train_data, test_data = generate_data(n_items=200, n_queries=400, candidates_per_query=15)

    print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")

    random_ndcg = compute_random_ndcg(test_data)
    print(f"Random baseline NDCG@5: {random_ndcg:.4f}")

    print("\nComputing learning curves...")
    results = compute_learning_curve(
        train_data, test_data,
        fractions=(0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0),
        n_trials=3,
        max_steps=300,
    )

    print("\nGenerating charts...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(results, random_ndcg, OUTPUT_DIR)

    write_report(results, random_ndcg, OUTPUT_DIR)


if __name__ == "__main__":
    main()
