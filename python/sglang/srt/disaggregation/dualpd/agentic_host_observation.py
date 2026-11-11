"""Opt-in, read-only Host arena counters owned by the serving engine.

The external experiment sampler consumes these byte records. No allocator lock,
tensor access, CUDA call, ownership transition or transfer progress occurs here.
Observation failures must not fail successful arena construction or cleanup.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
import weakref

logger = logging.getLogger(__name__)
_arenas = {}
_thread_pid = None
_registration_lock = threading.Lock()


def _after_fork_child():
    global _arenas, _registration_lock, _thread_pid
    _arenas = {}
    _registration_lock = threading.Lock()
    _thread_pid = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _argument(name):
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def snapshot():
    pools, errors = [], []
    for key, (reference, metadata) in tuple(_arenas.items()):
        arena = reference()
        if arena is None:
            if _arenas.get(key, (None,))[0] is reference:
                _arenas.pop(key, None)
            continue
        try:
            used, capacity = int(arena.used_bytes), int(arena.capacity_bytes)
            if not 0 <= used <= capacity:
                raise ValueError(f"inconsistent counter sample: used={used}, capacity={capacity}")
            pools.append({**metadata, "used_bytes": used, "capacity_bytes": capacity,
                          "committed_bytes": getattr(arena, "committed_bytes", None),
                          "closed": bool(getattr(arena, "_closed", False))})
        except Exception as error:
            errors.append(f"{key}: {type(error).__name__}: {error}")
    return dict(ts=time.time(), monotonic=time.monotonic(), pid=os.getpid(),
                role=_argument("--disaggregation-mode"), port=_argument("--port"),
                engine=os.environ.get("SGLANG_AGENTIC_KV_ENGINE_ID"), pools=pools, errors=errors)


def _write_samples(directory, interval):
    pid = os.getpid()
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        latest = directory / f"{pid}.latest.json"
        temporary = directory / f"{pid}.latest.tmp"
        with (directory / f"{pid}.jsonl").open("a", buffering=1) as stream:
            while os.getpid() == pid:
                try:
                    payload = json.dumps(snapshot(), separators=(",", ":")) + "\n"
                    stream.write(payload)
                    temporary.write_text(payload)
                    os.replace(temporary, latest)
                except Exception as error:
                    logger.warning("Host observation sample failed: %s", error)
                time.sleep(interval)
    except Exception as error:
        logger.warning("Host observation writer stopped: %s", error)


def register_arena(arena, direction):
    """Observe a constructed arena by weak reference, only when requested."""
    global _thread_pid
    directory = os.environ.get("PD_DRAM_TRACE_DIR")
    if not directory:
        return
    try:
        interval = float(os.environ.get("PD_DRAM_SAMPLE_SECONDS", "2"))
        if not 0 < interval < float("inf"):
            raise ValueError("Host observation interval must be finite and positive")
        if direction not in {"d2p", "p2d"}:
            raise ValueError("unknown Host observation direction")
        metadata = dict(kind="arena", direction=direction,
                        class_name=type(arena).__name__, object_id=id(arena),
                        backing=getattr(arena, "path", None))
        # This is an observer-only lock, never an allocator/ownership lock.
        with _registration_lock:
            if _thread_pid is not None and _thread_pid != os.getpid():
                _arenas.clear()  # Do not count inherited parent registrations twice.
            _arenas[id(arena)] = (weakref.ref(arena), metadata)
            if _thread_pid != os.getpid():
                threading.Thread(target=_write_samples, args=(directory, interval),
                                 name="dram-observer", daemon=True).start()
                _thread_pid = os.getpid()
    except Exception as error:
        logger.warning("Host observation registration failed: %s", error)
