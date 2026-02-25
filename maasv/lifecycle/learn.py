"""
Learn Job - Sleep worker handler for learned ranker training.

Three phases per idle cycle:
1. Label unlabeled retrieval logs (compute outcomes from re-access patterns)
2. Train model (gradient steps on labeled data)
3. Check graduation readiness (auto-graduate if enabled and criteria met)
"""

import logging
from typing import Callable

logger = logging.getLogger("maasv.lifecycle.learn")


def run_learn_job(data: dict, cancel_check: Callable[[], bool]):
    """
    Run a learn cycle during idle time.

    Phase 1: Label up to 100 retrieval log entries with outcomes.
    Phase 2: Train the ranking model for up to 50 gradient steps.
    Phase 3: Check if learned ranker is ready to graduate from shadow mode.
    """
    import maasv

    config = maasv.get_config()

    if not config.learned_ranker_enabled:
        logger.debug("[Learn] Learned ranker disabled, skipping")
        return

    from maasv.core.learned_ranker import (
        check_graduation_readiness,
        graduate_from_shadow_mode,
        label_outcomes,
        train,
    )

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

    # Phase 3: Check graduation readiness
    if cancel_check():
        return

    if not config.learned_ranker_shadow_mode:
        return  # Already graduated

    readiness = check_graduation_readiness()
    if readiness is None:
        return

    if readiness["ready"]:
        if config.learned_ranker_auto_graduate:
            graduated = graduate_from_shadow_mode()
            if graduated:
                logger.info(
                    f"[Learn] Auto-graduated learned ranker from shadow mode "
                    f"(ndcg={readiness['ndcg']:.3f}, avg_tau={readiness['avg_tau']:.2f}, "
                    f"tau_std={readiness['tau_std']:.2f}, "
                    f"comparisons={readiness['comparisons']})"
                )
        else:
            logger.info(
                f"[Learn] Learned ranker ready to graduate from shadow mode "
                f"(ndcg={readiness['ndcg']:.3f}, avg_tau={readiness['avg_tau']:.2f}, "
                f"tau_std={readiness['tau_std']:.2f}, "
                f"comparisons={readiness['comparisons']}). "
                f"Set learned_ranker_auto_graduate=True or call graduate_from_shadow_mode()."
            )
    else:
        logger.debug(
            f"[Learn] Graduation not ready: {readiness['reason']} (comparisons={readiness.get('comparisons', 0)})"
        )
