"""
Learn Job - Sleep worker handler for learned ranker training.

Two phases per idle cycle:
1. Label unlabeled retrieval logs (compute outcomes from re-access patterns)
2. Train model (gradient steps on labeled data)
"""

import logging
from typing import Callable

logger = logging.getLogger("maasv.lifecycle.learn")


def run_learn_job(data: dict, cancel_check: Callable[[], bool]):
    """
    Run a learn cycle during idle time.

    Phase 1: Label up to 100 retrieval log entries with outcomes.
    Phase 2: Train the ranking model for up to 50 gradient steps.
    """
    import maasv
    config = maasv.get_config()

    if not config.learned_ranker_enabled:
        logger.debug("[Learn] Learned ranker disabled, skipping")
        return

    from maasv.core.learned_ranker import label_outcomes, train

    # Phase 1: Label outcomes
    if cancel_check():
        return

    labeled = label_outcomes(
        cancel_check=cancel_check,
        max_entries=100,
    )

    if labeled > 0:
        logger.info(f"[Learn] Labeled {labeled} retrieval log entries")

    # Phase 2: Train model
    if cancel_check():
        return

    stats = train(
        cancel_check=cancel_check,
        max_steps=config.learned_ranker_max_steps,
        lr=config.learned_ranker_lr,
    )

    if stats:
        logger.info(
            f"[Learn] Training complete: {stats['training_samples']} samples, "
            f"{stats['steps']} steps, loss={stats.get('final_loss', 'N/A')}"
        )
    else:
        logger.debug("[Learn] Training skipped (insufficient labeled data)")
