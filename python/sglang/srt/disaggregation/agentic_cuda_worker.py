"""Bind dedicated background workers to their owning rank's CUDA device."""
import logging


def run_rank_bound_worker(device_id, callback, *args):
    # CUDA's current device is thread-local, not inherited from the scheduler.
    # In particular remote-key imports must not select the first active device
    # merely because the dedicated worker has no current CUDA context yet.
    import torch

    try:
        torch.cuda.set_device(device_id)
        return callback(*args)
    except Exception:
        logging.getLogger(__name__).exception(
            "Agentic CUDA worker failed device=%s callback=%s", device_id, callback
        )
        raise
