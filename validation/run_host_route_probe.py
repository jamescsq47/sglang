"""Diagnostic-only D capacity fault injection; no serving code is modified.

Use an otherwise idle P/D deployment. A separate router reports D as full
until its first native P2D Host offer, then restores real capacity after 3 s.
This exercises the real Host handoff without allocating artificial GPU KV.
"""
import argparse
import dataclasses
import logging
import os
import signal
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-pid", type=int, required=True)
    parser.add_argument("--port", type=int, default=29413)
    parser.add_argument("--prometheus-port", type=int, default=29414)
    args = parser.parse_args()
    proc = Path(f"/proc/{args.router_pid}")
    command = proc.joinpath("cmdline").read_bytes().split(b"\0")
    command = [part.decode() for part in command if part]
    assert command[1].endswith("launch_late_binding_router.py"), command[:2]
    env = dict(part.decode().split("=", 1) for part in proc.joinpath("environ").read_bytes().split(b"\0") if part)
    # Reuse exactly the deployment's control ledger/TP/Host configuration.
    os.environ.update(env)
    argv = command[2:]
    for flag, value in (("--port", str(args.port)), ("--prometheus-port", str(args.prometheus_port)), ("--host", "127.0.0.1")):
        argv[argv.index(flag) + 1] = value

    import late_binding_router
    import launch_late_binding_router

    class CapacityProbe(late_binding_router.LateBindingMiniLoadBalancer):
        _release_at = None
        _probe_started = None
        _initial_offers = None

        async def _refresh_decode_loads(self, session):
            loads = await super()._refresh_decode_loads(session)
            now = time.monotonic()
            if self._probe_started is None:
                self._probe_started = now
                self._initial_offers = set(self._p2d_host_offered_snapshots)
            if now - self._probe_started > 90 and self._release_at is None:
                self._release_at = now
                logging.error("HOST_PROBE expired without native Host offer; coverage FAILED")
            if self._p2d_host_offered_snapshots - self._initial_offers and self._release_at is None:
                self._release_at = now + 3.0
                logging.warning("HOST_PROBE native Host offer observed; restoring real capacity in 3 seconds")
            if self._release_at is None or now < self._release_at:
                loads = [dataclasses.replace(load, used_tokens=load.capacity_tokens,
                    physical_used_tokens=load.capacity_tokens) for load in loads]
                self._load_cache = loads
            return loads

    launch_late_binding_router.LateBindingMiniLoadBalancer = CapacityProbe
    sys.argv = [command[1], *argv]
    import requests
    for flag in ("--prefill", "--decode"):
        url = argv[argv.index(flag) + 1]
        response = requests.get(url + "/get_load", timeout=5)
        response.raise_for_status()
        assert all(row["num_reqs"] == 0 for row in response.json()), "Probe requires idle workers"
    # Freeze only this diagnostic deployment's idle original router so its
    # background pressure writer cannot race the probe. Supervisor stays up.
    os.kill(args.router_pid, signal.SIGSTOP)
    def restore_on_signal(signum, frame):
        os.kill(args.router_pid, signal.SIGCONT)
        raise SystemExit(128 + signum)
    # Uvicorn re-raises its captured signal on shutdown; restore the original
    # router before that re-raise can take Python past the outer finally.
    signal.signal(signal.SIGTERM, restore_on_signal)
    signal.signal(signal.SIGINT, restore_on_signal)
    try:
        launch_late_binding_router.main()
    finally:
        os.kill(args.router_pid, signal.SIGCONT)


if __name__ == "__main__":
    main()
