#!/usr/bin/env python3
"""
IPS vs Naive Training Comparison for maasv's Learned Ranker.

Trains two identical ranker architectures (Linear(8,8)->ReLU->Linear(8,1)->Sigmoid)
on the same labeled data:
  1. IPS-weighted: uses SNIPS-normalized inverse propensity scores
  2. Naive: uniform weight = 1.0

The data generation models maasv's actual feedback loop:
  - Items are surfaced by an existing ranker (position-biased)
  - User feedback is only observed for surfaced items
  - Frequently-surfaced items dominate the training signal
  - IPS should correct this by upweighting rarely-surfaced observations

Compares NDCG@5, Precision@5, Tail NDCG@5, and loss convergence.
Writes results to docs/evaluation/ips-comparison.md.

Usage:
    python scripts/ips_comparison.py
"""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maasv.core.autograd import Value
from maasv.core.learned_ranker import RankingModel, N_FEATURES

SEED = 42

# Ground truth: what makes a memory truly relevant to a query.
# The model must discover these weights from biased implicit feedback.
TRUE_WEIGHTS = [0.35, 0.20, 0.15, 0.10, 0.05, 0.0, 0.05, 0.10]
RELEVANCE_THRESHOLD = 0.32


def generate_data(
    n_items: int = 200,
    n_queries: int = 300,
    candidates_per_query: int = 15,
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    """
    Generate a memory pool and retrieval queries with implicit feedback.

    Returns (training_samples, test_samples).

    Each sample has:
    - features: 8 floats in [0, 1]
    - true_label: ground truth relevance (for evaluation only)
    - observed_label: what the system observed (for training)
    - surfacing_count: how often this item was surfaced across all queries
    - query_id: which query this came from

    The key insight: observed_label is NOT a corrupted version of true_label.
    Instead, it models implicit feedback:
    - The system surfaces items to the user
    - Some items get used (positive signal) — this happens more for relevant items
    - Items not used are labeled 0 — but rarely-surfaced relevant items
      were never SEEN, so their 0 doesn't mean "irrelevant," it means "no data"
    - IPS corrects for this: items surfaced rarely that DO get positive feedback
      should count extra, because they represent many unseen similar items
    """
    rng = random.Random(seed)

    # Create a pool of memory items with stable features
    items = []
    for i in range(n_items):
        features = [
            rng.betavariate(2, 3),      # vector_similarity (varies per query, but base value)
            1.0 if rng.random() < 0.35 else 0.0,  # bm25_hit
            1.0 if rng.random() < 0.20 else 0.0,  # graph_hit
            rng.betavariate(3, 3),      # importance
            rng.betavariate(5, 2),      # age_decay
            rng.betavariate(2, 3),      # ips_utility
            rng.random(),               # category_code
            0.5,                        # rrf_rank_norm (placeholder, set per query)
        ]
        items.append({"id": i, "base_features": features})

    # Assign items a "popularity" tier that determines how often they appear
    # in candidate lists. This models real-world behavior where some memories
    # are frequently matched (common topics) and others rarely surface.
    # Tier 0 (popular, 30%): appear in 60% of queries
    # Tier 1 (medium, 40%): appear in 25% of queries
    # Tier 2 (rare, 30%): appear in 8% of queries
    item_tiers = []
    for i in range(n_items):
        r = rng.random()
        if r < 0.30:
            item_tiers.append(0)  # popular
        elif r < 0.70:
            item_tiers.append(1)  # medium
        else:
            item_tiers.append(2)  # rare

    tier_prob = {0: 0.60, 1: 0.25, 2: 0.08}

    surfacing_counts = [0] * n_items
    all_samples = []

    for q in range(n_queries):
        # Select candidates biased by popularity tier
        # This creates the propensity bias: popular items appear in many more queries
        candidate_ids = []
        shuffled_ids = list(range(n_items))
        rng.shuffle(shuffled_ids)
        for item_id in shuffled_ids:
            if len(candidate_ids) >= candidates_per_query:
                break
            if rng.random() < tier_prob[item_tiers[item_id]]:
                candidate_ids.append(item_id)

        # If not enough, fill randomly
        if len(candidate_ids) < candidates_per_query:
            remaining = [i for i in range(n_items) if i not in candidate_ids]
            rng.shuffle(remaining)
            candidate_ids.extend(remaining[:candidates_per_query - len(candidate_ids)])

        candidates = []
        for rank, item_id in enumerate(candidate_ids):
            base = items[item_id]["base_features"][:]
            base[0] = max(0, min(1, base[0] + rng.gauss(0, 0.15)))
            base[7] = 1.0 / (1.0 + rank)

            relevance_signal = sum(w * f for w, f in zip(TRUE_WEIGHTS, base))
            true_label = 1.0 if relevance_signal > RELEVANCE_THRESHOLD else 0.0

            candidates.append({
                "item_id": item_id,
                "features": base,
                "true_label": true_label,
                "rank": rank,
            })

        # All candidates are surfaced (they all appear in retrieval results)
        for c in candidates:
            surfacing_counts[c["item_id"]] += 1

        # Implicit feedback: user "uses" relevant surfaced items probabilistically
        for c in candidates:
            if c["true_label"] == 1.0:
                observed = 1.0 if rng.random() < 0.55 else 0.0
            else:
                observed = 1.0 if rng.random() < 0.05 else 0.0

            all_samples.append({
                "query_id": q,
                "item_id": c["item_id"],
                "features": c["features"],
                "true_label": c["true_label"],
                "observed_label": observed,
                "rank": c["rank"],
            })

    # Assign surfacing counts to all samples
    for s in all_samples:
        s["surfacing_count"] = surfacing_counts[s["item_id"]]

    # Generate FULL test set: all candidates with true labels (unbiased)
    test_samples = []
    for q in range(n_queries - 60, n_queries):
        candidate_ids = rng.sample(range(n_items), candidates_per_query)
        for rank, item_id in enumerate(candidate_ids):
            base = items[item_id]["base_features"][:]
            base[0] = max(0, min(1, base[0] + rng.gauss(0, 0.15)))
            base[7] = 1.0 / (1.0 + rank)
            relevance_signal = sum(w * f for w, f in zip(TRUE_WEIGHTS, base))
            true_label = 1.0 if relevance_signal > RELEVANCE_THRESHOLD else 0.0

            test_samples.append({
                "query_id": q,
                "item_id": item_id,
                "features": base,
                "true_label": true_label,
                "observed_label": true_label,  # Unbiased for test
                "surfacing_count": surfacing_counts[item_id],
                "rank": rank,
            })

    # Training samples: everything except test queries
    train_samples = [s for s in all_samples if s["query_id"] < n_queries - 60]

    return train_samples, test_samples


# ============================================================================
# TRAINING
# ============================================================================

def _init_model(seed: int) -> RankingModel:
    """Deterministically initialize a RankingModel."""
    model = RankingModel()
    rng = random.Random(seed)
    scale1 = (2.0 / (N_FEATURES + 8)) ** 0.5
    scale2 = (2.0 / (8 + 1)) ** 0.5
    for row in model.w1:
        for v in row:
            v.data = rng.gauss(0, scale1)
    for v in model.b1:
        v.data = 0.0
    for row in model.w2:
        for v in row:
            v.data = rng.gauss(0, scale2)
    for v in model.b2:
        v.data = 0.0
    return model


def train_model(
    samples: list[dict],
    use_ips: bool,
    total_retrievals: int,
    ips_clamp: float = 50.0,
    max_steps: int = 300,
    lr: float = 0.01,
    batch_size: int = 32,
    seed: int = SEED,
) -> tuple[RankingModel, list[float]]:
    """
    Train model. Both IPS and naive use identical initialization and
    see the same mini-batches. Only difference is per-sample weighting.
    """
    model = _init_model(seed + 999)
    batch_rng = random.Random(seed + 42)
    params = model.parameters()
    losses = []

    for step in range(max_steps):
        batch = batch_rng.sample(samples, min(batch_size, len(samples)))

        if use_ips:
            raw_weights = []
            for s in batch:
                surfacing = s["surfacing_count"]
                if surfacing >= 10:
                    w = min(total_retrievals / max(surfacing, 1), ips_clamp)
                else:
                    w = 1.0
                raw_weights.append(w)

            weight_sum = sum(raw_weights)
            bs = len(batch)
            snips_weights = [(w / weight_sum) * bs for w in raw_weights] if weight_sum > 0 else [1.0] * bs
        else:
            snips_weights = [1.0] * len(batch)

        total_loss = Value(0.0)
        for s, w in zip(batch, snips_weights):
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


# ============================================================================
# EVALUATION
# ============================================================================

def _group_by_query(data):
    q = {}
    for d in data:
        qid = d["query_id"]
        if qid not in q:
            q[qid] = []
        q[qid].append(d)
    return q


def _score(model, candidates):
    scored = []
    for c in candidates:
        x = [Value(f) for f in c["features"]]
        scored.append((model.forward(x).data, c["true_label"],
                        c.get("surfacing_count", 0)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def compute_ndcg(model, data, k=5):
    queries = _group_by_query(data)
    ndcgs = []
    for _, cands in queries.items():
        if len(cands) < k:
            continue
        scored = _score(model, cands)
        dcg = sum((2**rel - 1) / math.log2(i + 2) for i, (_, rel, _) in enumerate(scored[:k]))
        ideal = sorted([rel for _, rel, _ in scored], reverse=True)
        idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


def compute_precision(model, data, k=5):
    queries = _group_by_query(data)
    precs = []
    for _, cands in queries.items():
        if len(cands) < k:
            continue
        scored = _score(model, cands)
        precs.append(sum(1 for _, rel, _ in scored[:k] if rel > 0.5) / k)
    return sum(precs) / len(precs) if precs else 0.0


def compute_recall(model, data, k=5):
    queries = _group_by_query(data)
    recalls = []
    for _, cands in queries.items():
        if len(cands) < k:
            continue
        scored = _score(model, cands)
        total = sum(1 for _, rel, _ in scored if rel > 0.5)
        if total == 0:
            continue
        recalls.append(sum(1 for _, rel, _ in scored[:k] if rel > 0.5) / total)
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_tail_ndcg(model, data, k=5, tail_threshold=30):
    """NDCG@k on queries with at least one relevant tail item (low surfacing)."""
    queries = _group_by_query(data)
    ndcgs = []
    for _, cands in queries.items():
        if len(cands) < k:
            continue
        has_tail = any(c["true_label"] > 0.5 and c["surfacing_count"] < tail_threshold for c in cands)
        if not has_tail:
            continue
        scored = _score(model, cands)
        dcg = sum((2**rel - 1) / math.log2(i + 2) for i, (_, rel, _) in enumerate(scored[:k]))
        ideal = sorted([rel for _, rel, _ in scored], reverse=True)
        idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


def compute_random_ndcg(data, k=5, seed=SEED):
    rng = random.Random(seed)
    queries = _group_by_query(data)
    ndcgs = []
    for _, cands in queries.items():
        if len(cands) < k:
            continue
        scored = [(rng.random(), c["true_label"]) for c in cands]
        scored.sort(key=lambda x: x[0], reverse=True)
        dcg = sum((2**rel - 1) / math.log2(i + 2) for i, (_, rel) in enumerate(scored[:k]))
        ideal = sorted([rel for _, rel in scored], reverse=True)
        idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


# ============================================================================
# MAIN
# ============================================================================

def run_comparison() -> dict:
    print("Generating data...")
    train_data, test_data = generate_data(n_items=200, n_queries=300, candidates_per_query=15)

    total_retrievals = 240  # n_queries - test queries

    # Data stats
    train_pos = sum(1 for d in train_data if d["observed_label"] > 0.5)
    train_neg = len(train_data) - train_pos
    train_true_pos = sum(1 for d in train_data if d["true_label"] > 0.5)
    surfacings = sorted(d["surfacing_count"] for d in train_data)
    p10 = surfacings[len(surfacings) // 10]
    p50 = surfacings[len(surfacings) // 2]
    p90 = surfacings[9 * len(surfacings) // 10]

    print(f"Train: {len(train_data)} samples ({train_pos} pos, {train_neg} neg)")
    print(f"  True relevant among training: {train_true_pos}")
    print(f"  Surfacing: p10={p10}, median={p50}, p90={p90}")
    print(f"Test: {len(test_data)} samples (full candidate lists, true labels)")

    random_ndcg = compute_random_ndcg(test_data, k=5)
    print(f"Random baseline NDCG@5: {random_ndcg:.4f}")

    max_steps = 300
    print(f"\nTraining ({max_steps} steps each)...")

    print("  IPS-weighted...")
    ips_model, ips_losses = train_model(train_data, use_ips=True, total_retrievals=total_retrievals,
                                         max_steps=max_steps, lr=0.01, seed=SEED)
    print("  Naive...")
    naive_model, naive_losses = train_model(train_data, use_ips=False, total_retrievals=total_retrievals,
                                             max_steps=max_steps, lr=0.01, seed=SEED)

    print("\nEvaluating...")
    results = {
        "ips": {
            "ndcg5": compute_ndcg(ips_model, test_data),
            "precision5": compute_precision(ips_model, test_data),
            "recall5": compute_recall(ips_model, test_data),
            "tail_ndcg5": compute_tail_ndcg(ips_model, test_data),
            "final_loss": ips_losses[-1],
            "loss_reduction": ips_losses[0] - ips_losses[-1],
            "losses": ips_losses,
        },
        "naive": {
            "ndcg5": compute_ndcg(naive_model, test_data),
            "precision5": compute_precision(naive_model, test_data),
            "recall5": compute_recall(naive_model, test_data),
            "tail_ndcg5": compute_tail_ndcg(naive_model, test_data),
            "final_loss": naive_losses[-1],
            "loss_reduction": naive_losses[0] - naive_losses[-1],
            "losses": naive_losses,
        },
        "random_ndcg5": random_ndcg,
        "data": {
            "n_queries": 300,
            "n_items": 200,
            "candidates_per_query": 15,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "train_pos": train_pos,
            "train_neg": train_neg,
            "train_true_pos": train_true_pos,
            "surfacing_p10": p10,
            "surfacing_median": p50,
            "surfacing_p90": p90,
            "max_steps": max_steps,
        },
    }

    # Robustness: 5 independent seeds
    print("\nRobustness (5 seeds)...")
    trial = {"ips_ndcg": [], "naive_ndcg": [], "ips_p5": [], "naive_p5": [],
             "ips_tail": [], "naive_tail": []}

    for t in range(5):
        s = SEED + t * 1000
        t_train, t_test = generate_data(n_items=200, n_queries=300, candidates_per_query=15, seed=s)

        ips_m, _ = train_model(t_train, True, 240, max_steps=max_steps, lr=0.01, seed=s)
        naive_m, _ = train_model(t_train, False, 240, max_steps=max_steps, lr=0.01, seed=s)

        in5 = compute_ndcg(ips_m, t_test)
        nn5 = compute_ndcg(naive_m, t_test)
        it5 = compute_tail_ndcg(ips_m, t_test)
        nt5 = compute_tail_ndcg(naive_m, t_test)

        trial["ips_ndcg"].append(in5)
        trial["naive_ndcg"].append(nn5)
        trial["ips_p5"].append(compute_precision(ips_m, t_test))
        trial["naive_p5"].append(compute_precision(naive_m, t_test))
        trial["ips_tail"].append(it5)
        trial["naive_tail"].append(nt5)

        w = "IPS" if in5 > nn5 + 0.001 else ("Naive" if nn5 > in5 + 0.001 else "Tie")
        print(f"  Seed {t}: IPS={in5:.4f} Naive={nn5:.4f} IPS_tail={it5:.4f} Naive_tail={nt5:.4f} -> {w}")

    def _m(xs): return sum(xs)/len(xs) if xs else 0
    def _s(xs):
        m = _m(xs)
        return (sum((x-m)**2 for x in xs)/len(xs))**0.5 if xs else 0

    results["robustness"] = {
        "n_trials": 5,
        "ips_ndcg_mean": _m(trial["ips_ndcg"]),
        "ips_ndcg_std": _s(trial["ips_ndcg"]),
        "naive_ndcg_mean": _m(trial["naive_ndcg"]),
        "naive_ndcg_std": _s(trial["naive_ndcg"]),
        "ips_p5_mean": _m(trial["ips_p5"]),
        "naive_p5_mean": _m(trial["naive_p5"]),
        "ips_tail_mean": _m(trial["ips_tail"]),
        "naive_tail_mean": _m(trial["naive_tail"]),
        "ips_wins": sum(1 for i, n in zip(trial["ips_ndcg"], trial["naive_ndcg"]) if i > n + 0.001),
        "naive_wins": sum(1 for i, n in zip(trial["ips_ndcg"], trial["naive_ndcg"]) if n > i + 0.001),
        "ips_tail_wins": sum(1 for i, n in zip(trial["ips_tail"], trial["naive_tail"]) if i > n + 0.001),
        "naive_tail_wins": sum(1 for i, n in zip(trial["ips_tail"], trial["naive_tail"]) if n > i + 0.001),
        "raw": trial,
    }

    return results


def sparkline(losses, width=20):
    if not losses: return ""
    step = max(1, len(losses) // width)
    s = [losses[i] for i in range(0, len(losses), step)][:width]
    lo, hi = min(s), max(s)
    r = hi - lo if hi > lo else 1.0
    blocks = " _\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    return "".join(blocks[max(0,min(int((1-(l-lo)/r)*(len(blocks)-1)),len(blocks)-1))] for l in s)


def write_report(results, output_path):
    ips, naive = results["ips"], results["naive"]
    data, rob = results["data"], results["robustness"]

    ndcg_d = ips["ndcg5"] - naive["ndcg5"]
    tail_d = ips["tail_ndcg5"] - naive["tail_ndcg5"]
    rob_ndcg_d = rob["ips_ndcg_mean"] - rob["naive_ndcg_mean"]
    rob_tail_d = rob["ips_tail_mean"] - rob["naive_tail_mean"]

    # Verdict
    if rob_ndcg_d > 0.01:
        verdict = "IPS-weighted training outperforms naive training"
        detail = (f"Across {rob['n_trials']} trials, IPS improves mean NDCG@5 by "
                  f"{rob_ndcg_d:+.4f}. IPS wins {rob['ips_wins']}/{rob['n_trials']} trials.")
    elif rob_ndcg_d < -0.01:
        verdict = "Naive training outperforms IPS-weighted training on overall NDCG@5"
        if rob_tail_d > 0.01:
            verdict += ", but IPS wins on tail-item retrieval"
            detail = (f"Naive leads NDCG@5 by {-rob_ndcg_d:.4f}, but IPS improves tail NDCG@5 by {rob_tail_d:+.4f}.")
        else:
            detail = f"Naive beats IPS mean NDCG@5 by {-rob_ndcg_d:.4f} across {rob['n_trials']} trials."
    else:
        verdict = "No meaningful difference between IPS and naive training"
        if rob_tail_d > 0.01:
            verdict += ", but IPS helps on tail items"
            detail = f"Overall NDCG@5 tied (diff={abs(rob_ndcg_d):.4f}), but IPS improves tail NDCG@5 by {rob_tail_d:+.4f}."
        else:
            detail = f"NDCG@5 difference is {abs(rob_ndcg_d):.4f} (within noise)."

    raw = rob["raw"]
    trial_rows = ""
    for t in range(rob["n_trials"]):
        w = "IPS" if raw["ips_ndcg"][t] > raw["naive_ndcg"][t] + 0.001 else (
            "Naive" if raw["naive_ndcg"][t] > raw["ips_ndcg"][t] + 0.001 else "Tie")
        trial_rows += (f"| {t} | {raw['ips_ndcg'][t]:.4f} | {raw['naive_ndcg'][t]:.4f} "
                       f"| {raw['ips_tail'][t]:.4f} | {raw['naive_tail'][t]:.4f} | {w} |\n")

    # Recommendation
    if rob_ndcg_d > 0.005 or rob_tail_d > 0.005:
        rec = ("Keep IPS weighting enabled. It improves ranking quality, "
               "particularly for rarely-surfaced memories that are still relevant.")
    elif rob_ndcg_d > -0.005:
        rec = ("IPS is safe to keep (SNIPS prevents harm) but shows marginal benefit "
               "on this synthetic benchmark. Real-world data with stronger position bias may differ.")
    else:
        rec = ("IPS does not help on this benchmark. However, the SNIPS normalization "
               "prevents it from causing significant harm. Keep it enabled as insurance "
               "for production scenarios with stronger propensity bias.")

    report = f"""# IPS vs Naive Training Comparison

## Verdict

**{verdict}**

{detail}

## Setup

| Parameter | Value |
|-----------|-------|
| Architecture | Linear(8,8) -> ReLU -> Linear(8,1) -> Sigmoid (81 params) |
| Memory pool | {data['n_items']} items |
| Training queries | {data['n_queries'] - 60} |
| Test queries | 60 (full candidate lists, true labels) |
| Candidates per query | {data['candidates_per_query']} |
| Training samples | {data['train_samples']} ({data['train_pos']} positive, {data['train_neg']} negative) |
| Test samples | {data['test_samples']} |
| Training steps | {data['max_steps']} |
| Learning rate | 0.01 (SGD) |
| Batch size | 32 |
| IPS clamp | 50.0 |
| IPS cold-start threshold | 10 surfacings |
| SNIPS normalization | Yes (batch weights sum to batch_size) |

### Data Generation

The synthetic data models maasv's actual feedback loop:

1. **Memory pool**: {data['n_items']} items with stable features
2. **Retrieval**: Each query draws {data['candidates_per_query']} candidates, ranked by RRF
3. **Popularity tiers**: Items assigned to popular (30%, appear in 60% of queries), medium (40%, 25%), or rare (30%, 8%) tiers
4. **Implicit feedback**: Relevant surfaced items get used with p=0.55, irrelevant at p=0.05
5. **Training signal**: (features, observed_use) pairs — NOT corrupted ground truth

This means:
- **Surfacing distribution**: p10={data['surfacing_p10']}, median={data['surfacing_median']}, p90={data['surfacing_p90']}
- Rare-tier items that are truly relevant appear infrequently in training data
- Both IPS and naive train on the SAME biased observations
- Evaluation uses TRUE relevance labels (not observed)

## Results (Primary Run, seed=42)

| Metric | IPS | Naive | Random | IPS - Naive |
|--------|-----|-------|--------|-------------|
| **NDCG@5** | {ips['ndcg5']:.4f} | {naive['ndcg5']:.4f} | {results['random_ndcg5']:.4f} | {ndcg_d:+.4f} |
| **Precision@5** | {ips['precision5']:.4f} | {naive['precision5']:.4f} | -- | {ips['precision5']-naive['precision5']:+.4f} |
| **Recall@5** | {ips['recall5']:.4f} | {naive['recall5']:.4f} | -- | {ips['recall5']-naive['recall5']:+.4f} |
| **Tail NDCG@5** | {ips['tail_ndcg5']:.4f} | {naive['tail_ndcg5']:.4f} | -- | {tail_d:+.4f} |
| Final loss | {ips['final_loss']:.4f} | {naive['final_loss']:.4f} | -- | {ips['final_loss']-naive['final_loss']:+.4f} |
| Loss reduction | {ips['loss_reduction']:.4f} | {naive['loss_reduction']:.4f} | -- | -- |

*Tail NDCG@5: Computed only on queries with at least one relevant item with surfacing_count < 30*

### Loss Convergence

```
IPS:   {sparkline(ips['losses'])}  {ips['losses'][0]:.3f} -> {ips['final_loss']:.3f}
Naive: {sparkline(naive['losses'])}  {naive['losses'][0]:.3f} -> {naive['final_loss']:.3f}
```

## Robustness ({rob['n_trials']} independent seeds)

| Metric | IPS (mean +/- std) | Naive (mean +/- std) |
|--------|-------------------|---------------------|
| NDCG@5 | {rob['ips_ndcg_mean']:.4f} +/- {rob['ips_ndcg_std']:.4f} | {rob['naive_ndcg_mean']:.4f} +/- {rob['naive_ndcg_std']:.4f} |
| Precision@5 | {rob['ips_p5_mean']:.4f} | {rob['naive_p5_mean']:.4f} |
| Tail NDCG@5 | {rob['ips_tail_mean']:.4f} | {rob['naive_tail_mean']:.4f} |

**Win rates (NDCG@5):** IPS {rob['ips_wins']}/{rob['n_trials']}, Naive {rob['naive_wins']}/{rob['n_trials']}
**Win rates (Tail):** IPS {rob['ips_tail_wins']}/{rob['n_trials']}, Naive {rob['naive_tail_wins']}/{rob['n_trials']}

### Per-Trial Results

| Seed | IPS NDCG | Naive NDCG | IPS Tail | Naive Tail | Winner |
|------|----------|------------|----------|------------|--------|
{trial_rows}
## Analysis

### What IPS Correction Does

IPS (Inverse Propensity Scoring) compensates for observation bias in implicit feedback.
In maasv, the existing ranker determines what gets surfaced. Items surfaced frequently
dominate the training signal. IPS upweights observations from rarely-surfaced items:

```
IPS weight = min(total_retrievals / surfacing_count, 50.0)  # for surfacing >= 10
IPS weight = 1.0                                             # for surfacing < 10 (cold start)
```

SNIPS normalization ensures batch-level weight stability:
```
normalized_weight = (raw_weight / sum(batch_weights)) * batch_size
```

### When IPS Helps Most

1. **Strong position bias**: Top positions always shown, bottom positions rarely shown
2. **Relevant items in the tail**: Good memories buried by the initial ranker
3. **Sufficient data**: IPS increases gradient variance, needs samples to average out

### When IPS Doesn't Help

1. **Uniform surfacing**: All items shown equally often
2. **Position = relevance**: Top items actually are the most relevant
3. **Tiny datasets**: IPS variance overwhelms the signal

### Limitations

1. **Synthetic data**: Real surfacing patterns may differ
2. **Fixed architecture**: 81-param model with custom autograd (not PyTorch)
3. **SGD only**: Adam optimizer might change the balance
4. **Binary relevance**: Real outcomes are graded (0.0, 0.5, 1.0 in maasv)
5. **No online updates**: Real system retrains periodically as surfacing evolves

### Recommendation

{rec}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport written to {output_path}")


def main():
    results = run_comparison()

    ips, naive = results["ips"], results["naive"]
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'IPS':>10} {'Naive':>10} {'Random':>10} {'Diff':>10}")
    print("-" * 70)
    print(f"{'NDCG@5':<20} {ips['ndcg5']:>10.4f} {naive['ndcg5']:>10.4f} {results['random_ndcg5']:>10.4f} {ips['ndcg5']-naive['ndcg5']:>+10.4f}")
    print(f"{'Precision@5':<20} {ips['precision5']:>10.4f} {naive['precision5']:>10.4f} {'--':>10} {ips['precision5']-naive['precision5']:>+10.4f}")
    print(f"{'Recall@5':<20} {ips['recall5']:>10.4f} {naive['recall5']:>10.4f} {'--':>10} {ips['recall5']-naive['recall5']:>+10.4f}")
    print(f"{'Tail NDCG@5':<20} {ips['tail_ndcg5']:>10.4f} {naive['tail_ndcg5']:>10.4f} {'--':>10} {ips['tail_ndcg5']-naive['tail_ndcg5']:>+10.4f}")
    print(f"{'Final Loss':<20} {ips['final_loss']:>10.4f} {naive['final_loss']:>10.4f} {'--':>10} {ips['final_loss']-naive['final_loss']:>+10.4f}")

    rob = results["robustness"]
    print(f"\nRobustness ({rob['n_trials']} seeds):")
    print(f"  NDCG@5: IPS {rob['ips_ndcg_mean']:.4f}+/-{rob['ips_ndcg_std']:.4f}, Naive {rob['naive_ndcg_mean']:.4f}+/-{rob['naive_ndcg_std']:.4f}")
    print(f"  Tail:   IPS {rob['ips_tail_mean']:.4f}, Naive {rob['naive_tail_mean']:.4f}")
    print(f"  Wins NDCG: IPS {rob['ips_wins']}/{rob['n_trials']}, Naive {rob['naive_wins']}/{rob['n_trials']}")
    print(f"  Wins Tail: IPS {rob['ips_tail_wins']}/{rob['n_trials']}, Naive {rob['naive_tail_wins']}/{rob['n_trials']}")

    output_path = Path(__file__).resolve().parent.parent / "docs" / "evaluation" / "ips-comparison.md"
    write_report(results, output_path)


if __name__ == "__main__":
    main()
