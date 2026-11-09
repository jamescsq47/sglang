"""Late concrete dispatch for native Scheduler integration, never an import."""

import sys


def _native_scheduler_class():
    module = sys.modules.get("sglang.srt.managers.scheduler")
    if module is None or not hasattr(module, "Scheduler"):
        raise RuntimeError("native Scheduler is not initialized")
    return module.Scheduler
