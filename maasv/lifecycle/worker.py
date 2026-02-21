"""
Sleep Worker - Background processing during idle periods.

Runs consolidation jobs when idle, cancels immediately
when activity resumes. Jobs are designed to be interruptible.
"""

import threading
import queue
import logging
import time
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

logger = logging.getLogger("maasv.lifecycle")


class JobType(Enum):
    INFERENCE = "inference"
    REVIEW = "review"
    REORGANIZE = "reorganize"
    MEMORY_HYGIENE = "memory_hygiene"
    LEARN = "learn"


@dataclass
class SleepJob:
    """A sleep-time compute job."""
    job_type: JobType
    data: dict
    priority: int = 0


class SleepWorker:
    """Background worker for sleep-time compute. Jobs are cancellable."""

    def __init__(self):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=50)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._started = False
        self._cancelled = threading.Event()
        self._job_counter = 0
        self._lock = threading.Lock()

    def _ensure_started(self):
        """Lazily start the worker thread on first job."""
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._started = True
            self._running = True
            self._cancelled.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="sleep-worker")
            self._thread.start()
            logger.info("[Sleep] Worker thread started")

    def stop(self):
        """Stop the worker thread."""
        if not self._running:
            return
        self._running = False
        self._cancelled.set()
        try:
            self._queue.put_nowait((0, 0, None))
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("[Sleep] Worker thread stopped")

    def cancel_current_work(self):
        """Cancel current job and clear queue."""
        with self._lock:
            self._cancelled.set()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        logger.debug("[Sleep] Cancelled current work")

    def resume_work(self):
        """Resume accepting and processing jobs."""
        with self._lock:
            self._cancelled.clear()
        logger.debug("[Sleep] Resumed work")

    def is_cancelled(self) -> bool:
        """Check if current work is cancelled."""
        return self._cancelled.is_set()

    def queue_job(self, job: SleepJob) -> bool:
        """Queue a sleep-time job (non-blocking). Returns True if queued."""
        if self._cancelled.is_set():
            return False
        self._ensure_started()
        with self._lock:
            self._job_counter += 1
            counter = self._job_counter
        try:
            self._queue.put_nowait((-job.priority, counter, job))
            logger.debug(f"[Sleep] Queued {job.job_type.value} job")
            return True
        except queue.Full:
            logger.warning("[Sleep] Queue full, dropping job")
            return False

    def _run(self):
        """Main worker loop."""
        logger.info("[Sleep] Worker loop starting")
        while self._running:
            try:
                priority, counter, job = self._queue.get(timeout=1.0)
                if job is None:
                    break
                if self._cancelled.is_set():
                    continue
                self._process_job(job)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[Sleep] Unexpected error: {e}", exc_info=True)
        logger.info("[Sleep] Worker loop exiting")

    def _process_job(self, job: SleepJob):
        """Process a single sleep job."""
        try:
            logger.info(f"[Sleep] Processing {job.job_type.value} job")
            start = time.time()

            if job.job_type == JobType.INFERENCE:
                from maasv.lifecycle.inference import run_inference_job
                run_inference_job(job.data, cancel_check=self.is_cancelled)
            elif job.job_type == JobType.REVIEW:
                from maasv.lifecycle.review import run_review_job
                run_review_job(job.data, cancel_check=self.is_cancelled)
            elif job.job_type == JobType.REORGANIZE:
                from maasv.lifecycle.reorganize import run_reorganize_job
                run_reorganize_job(job.data, cancel_check=self.is_cancelled)
            elif job.job_type == JobType.MEMORY_HYGIENE:
                from maasv.lifecycle.memory_hygiene import run_memory_hygiene_job
                run_memory_hygiene_job(job.data, cancel_check=self.is_cancelled)
            elif job.job_type == JobType.LEARN:
                from maasv.lifecycle.learn import run_learn_job
                run_learn_job(job.data, cancel_check=self.is_cancelled)

            elapsed = time.time() - start
            logger.info(f"[Sleep] Completed {job.job_type.value} job in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"[Sleep] Job failed: {e}", exc_info=True)


# Singleton
_worker: Optional[SleepWorker] = None
_worker_lock = threading.Lock()


def get_sleep_worker() -> SleepWorker:
    """Get the global sleep worker (creates if needed)."""
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = SleepWorker()
    return _worker


# Idle monitoring
_idle_monitor_thread: Optional[threading.Thread] = None
_idle_monitor_running = False
_idle_monitor_lock = threading.Lock()

# Default thresholds — overridden from config at runtime
_IDLE_THRESHOLD = 30
_IDLE_CHECK_INTERVAL = 5


def start_idle_monitor(
    get_last_activity: Callable[[], float],
    on_idle: Callable[[], None],
    on_active: Callable[[], None]
):
    """Start the idle monitor thread."""
    global _idle_monitor_thread, _idle_monitor_running, _IDLE_THRESHOLD, _IDLE_CHECK_INTERVAL

    with _idle_monitor_lock:
        if _idle_monitor_running:
            return

        # Read thresholds from config if initialized
        try:
            import maasv
            config = maasv.get_config()
            _IDLE_THRESHOLD = config.idle_threshold_seconds
            _IDLE_CHECK_INTERVAL = config.idle_check_interval
        except RuntimeError:
            pass

        _idle_monitor_running = True

        def monitor_loop():
            was_idle = False
            while _idle_monitor_running:
                try:
                    time.sleep(_IDLE_CHECK_INTERVAL)
                    if not _idle_monitor_running:
                        break

                    last_activity = get_last_activity()
                    idle_duration = time.time() - last_activity
                    is_idle = idle_duration >= _IDLE_THRESHOLD

                    if is_idle and not was_idle:
                        logger.info(f"[Sleep] Session idle for {idle_duration:.0f}s, starting sleep work")
                        on_idle()
                    elif not is_idle and was_idle:
                        logger.info("[Sleep] Session active, cancelling sleep work")
                        on_active()

                    was_idle = is_idle

                except Exception as e:
                    logger.error(f"[Sleep] Monitor error: {e}", exc_info=True)

        _idle_monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name="idle-monitor")
        _idle_monitor_thread.start()
        logger.info("[Sleep] Idle monitor started")


def stop_idle_monitor():
    """Stop the idle monitor thread."""
    global _idle_monitor_running
    with _idle_monitor_lock:
        _idle_monitor_running = False
    logger.info("[Sleep] Idle monitor stopped")
