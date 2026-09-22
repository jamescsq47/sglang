"""Run-owned DRAM control broker. No KV bytes, CUDA or control files.

Do not restart this broker under an existing run ID after a failure. Workers
must quiesce all physical transfers before a new run allocates their resources.
This entry point is not an engine integration/experiment acceptance gate.
"""

import argparse
import json
import os
import signal
import threading

from sglang.srt.disaggregation.agentic_control_rpc import ControlRPCServer
from sglang.srt.disaggregation.agentic_control_store import MemoryControlStore
from sglang.srt.disaggregation.agentic_host_rpc import HostLedgerService
from sglang.srt.disaggregation.agentic_remote_host_control import RemoteHostControlState


def create_server(run_id, token, address, *, tp_server=None):
    server = ControlRPCServer(
        run_id, token, address, max_records=1_000_000, max_namespaces=1024,
        start=False,
    )
    try:
        records = MemoryControlStore(server.publish)
        server.register_service("records", records.methods())
        host_services = {direction: HostLedgerService(server, direction)
                         for direction in ("d2p", "p2d")}
        remote = RemoteHostControlState(
            ledger_lookup=lambda direction, sid: host_services[direction].ledger.get(sid),
        )
        server.register_service("remote_host", {"call": remote.handle})
        system_methods = {"describe": lambda: {
            "run_id": run_id, "backend": "tcp-memory", "protocol": 1,
        }}
        if tp_server is not None:
            system_methods["tp_stats"] = tp_server.stats
        server.register_service("system", system_methods)
        server.start()
        return server
    except BaseException:
        server.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tp-port", type=int, required=True)
    parser.add_argument("--prefill-group", action="append", required=True)
    parser.add_argument("--decode-group", action="append", required=True)
    args = parser.parse_args()
    token = os.environ.get("SGLANG_AGENTIC_CONTROL_TOKEN", "")
    if not token:
        parser.error("SGLANG_AGENTIC_CONTROL_TOKEN is required; never pass it in argv")
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    from sglang.srt.disaggregation.agentic_tp_events import TPEventServer
    tp_server = TPEventServer(
        args.run_id, token, address=(args.listen, args.tp_port),
        # Generic lifecycle tombstones intentionally survive for the run so a
        # delayed TP report cannot revive an old request-generation.  A 500
        # episode agent workload produces roughly half a million bounded
        # entries; match the record service's production capacity rather than
        # inheriting the smaller unit-test/default server bound.
        max_entries=1_000_000,
        receipt_observers={(group, "p2d-receiver"): args.prefill_group
                           for group in args.decode_group},
    )
    server = None
    try:
        server = create_server(args.run_id, token, (args.listen, args.port), tp_server=tp_server)
        print(json.dumps({"control_ready": True, "run_id": args.run_id,
                          "address": server.address, "tp_address": tp_server.address}), flush=True)
        stopped.wait()
    finally:
        if server is not None:
            server.close()
        tp_server.close()


if __name__ == "__main__":
    main()
