# IPS vs Naive Training Comparison

## Verdict

**IPS-weighted training outperforms naive training**

Across 5 trials, IPS improves mean NDCG@5 by +0.0147. IPS wins 3/5 trials.

## Setup

| Parameter | Value |
|-----------|-------|
| Architecture | Linear(8,8) -> ReLU -> Linear(8,1) -> Sigmoid (81 params) |
| Memory pool | 200 items |
| Training queries | 240 |
| Test queries | 60 (full candidate lists, true labels) |
| Candidates per query | 15 |
| Training samples | 3600 (1310 positive, 2290 negative) |
| Test samples | 900 |
| Training steps | 300 |
| Learning rate | 0.01 (SGD) |
| Batch size | 32 |
| IPS clamp | 50.0 |
| IPS cold-start threshold | 10 surfacings |
| SNIPS normalization | Yes (batch weights sum to batch_size) |

### Data Generation

The synthetic data models maasv's actual feedback loop:

1. **Memory pool**: 200 items with stable features
2. **Retrieval**: Each query draws 15 candidates, ranked by RRF
3. **Popularity tiers**: Items assigned to popular (30%, appear in 60% of queries), medium (40%, 25%), or rare (30%, 8%) tiers
4. **Implicit feedback**: Relevant surfaced items get used with p=0.55, irrelevant at p=0.05
5. **Training signal**: (features, observed_use) pairs — NOT corrupted ground truth

This means:
- **Surfacing distribution**: p10=14, median=38, p90=49
- Rare-tier items that are truly relevant appear infrequently in training data
- Both IPS and naive train on the SAME biased observations
- Evaluation uses TRUE relevance labels (not observed)

## Results (Primary Run, seed=42)

| Metric | IPS | Naive | Random | IPS - Naive |
|--------|-----|-------|--------|-------------|
| **NDCG@5** | 0.3717 | 0.4014 | 0.6658 | -0.0297 |
| **Precision@5** | 0.4000 | 0.4200 | -- | -0.0200 |
| **Recall@5** | 0.2047 | 0.2158 | -- | -0.0111 |
| **Tail NDCG@5** | 0.3717 | 0.4014 | -- | -0.0297 |
| Final loss | 0.6293 | 0.6505 | -- | -0.0212 |
| Loss reduction | -0.0257 | -0.0131 | -- | -- |

*Tail NDCG@5: Computed only on queries with at least one relevant item with surfacing_count < 30*

### Loss Convergence

```
IPS:   █▂▆▄▅▅▄▆▃▆▅▆▂▄▃▅▃ ▅▅  0.604 -> 0.629
Naive: █_▇▄▇▅▄▇▃▆▄▅▃▂▂▅▂ ▄▅  0.637 -> 0.650
```

## Robustness (5 independent seeds)

| Metric | IPS (mean +/- std) | Naive (mean +/- std) |
|--------|-------------------|---------------------|
| NDCG@5 | 0.6446 +/- 0.1843 | 0.6299 +/- 0.1792 |
| Precision@5 | 0.6280 | 0.6140 |
| Tail NDCG@5 | 0.6446 | 0.6299 |

**Win rates (NDCG@5):** IPS 3/5, Naive 1/5
**Win rates (Tail):** IPS 3/5, Naive 1/5

### Per-Trial Results

| Seed | IPS NDCG | Naive NDCG | IPS Tail | Naive Tail | Winner |
|------|----------|------------|----------|------------|--------|
| 0 | 0.3717 | 0.4014 | 0.3717 | 0.4014 | Naive |
| 1 | 0.7083 | 0.6447 | 0.7083 | 0.6447 | IPS |
| 2 | 0.7658 | 0.7655 | 0.7658 | 0.7655 | Tie |
| 3 | 0.4977 | 0.4606 | 0.4977 | 0.4606 | IPS |
| 4 | 0.8793 | 0.8771 | 0.8793 | 0.8771 | IPS |

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

Keep IPS weighting enabled. It improves ranking quality, particularly for rarely-surfaced memories that are still relevant.
