"""Format benchmark JSON results as markdown tables.

Usage:
    python -m benchmarks.format_results [results/retrieval_quality_42.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def format_markdown(data: dict) -> str:
    """Convert benchmark JSON to a markdown table."""
    meta = data.get("meta", {})
    limit = meta.get("limit", 5)
    lines = []

    lines.append(f"## Retrieval Quality (seed={meta.get('seed')}, "
                 f"scale={meta.get('scale')}, "
                 f"k={limit})")
    lines.append("")
    lines.append(f"Dataset: {meta.get('num_memories', '?')} memories, "
                 f"{meta.get('num_queries', '?')} queries")
    lines.append("")
    lines.append(f"| Adapter | NDCG@{limit} | MRR | P@{limit} | Setup (s) | Eval (s) |")
    lines.append("|---------|--------|-----|------|-----------|----------|")

    for name, adapter_data in data.get("adapters", {}).items():
        if "error" in adapter_data:
            lines.append(f"| {name} | ERROR | — | — | — | — |")
            continue
        agg = adapter_data.get("aggregate", {})
        timing = adapter_data.get("timing", {})
        lines.append(
            f"| {name} "
            f"| {agg.get('ndcg_mean', 0):.4f} "
            f"| {agg.get('mrr_mean', 0):.4f} "
            f"| {agg.get('precision_mean', 0):.4f} "
            f"| {timing.get('setup_s', '—')} "
            f"| {timing.get('eval_s', '—')} |"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    results_dir = Path(__file__).parent / "results"

    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(results_dir.glob("retrieval_quality_*.json"))

    if not paths:
        print("No result files found.", file=sys.stderr)
        return 1

    for path in paths:
        with open(path) as f:
            data = json.load(f)
        print(format_markdown(data))

    return 0


if __name__ == "__main__":
    sys.exit(main())
