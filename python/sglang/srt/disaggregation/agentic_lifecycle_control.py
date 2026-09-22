"""File-free lifecycle fences and bounded background metadata completion.

"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

from sglang.srt.disaggregation.agentic_control_store import control_kv


def lifecycle_records():
    return control_kv("lifecycle-fences", os.getenv("SGLANG_PD_P_READY_DIR", "default"))


_executor = None
_lock = threading.Lock()
_slots = threading.BoundedSemaphore(256)


def poll_lifecycle_call(holder, key, function, *args, **kwargs):
    """One retained Future per caller phase; never block a scheduler tick.

    Errors stay attached to the phase: an ambiguous ownership result must not
    cause another physical allocation or discard an in-flight workset. Caller
    state retirement drops the completed Future together with its lease.
    """
    global _executor
    with _lock:
        pending = getattr(holder, "_agentic_lifecycle_futures", None)
        if pending is None:
            pending = {}
            holder._agentic_lifecycle_futures = pending
        future = pending.get(key)
        if future is None:
            if not _slots.acquire(blocking=False):
                return False, None
            if _executor is None:
                _executor = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="kv-lifecycle"
                )
            try:
                future = _executor.submit(function, *args, **kwargs)
            except BaseException:
                _slots.release()
                raise
            future.add_done_callback(lambda _: _slots.release())
            pending[key] = future
    if not future.done():
        return False, None
    return True, future.result()
