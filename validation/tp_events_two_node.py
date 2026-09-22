"""CPU protocol smoke, NOT a KV transfer or TP model performance result."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shlex
import socket
import subprocess
import sys
import time


# The remote host receives source through stdin and evaluates it in memory.
# This avoids depending on NFS visibility of a newly edited module on the peer.
REMOTE_BOOTSTRAP = """
import json, sys, types
payload = json.load(sys.stdin)
module = types.ModuleType('dualpd_tp_smoke_protocol')
sys.modules[module.__name__] = module
exec(compile(payload['source'], '<tp-event-protocol>', 'exec'), module.__dict__)
exec(compile(payload['worker'], '<tp-event-smoke-worker>', 'exec'))
"""

REMOTE_WORKER = """
import multiprocessing, time, traceback

cfg = payload['config']

def wait(client, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        client.changed.clear()
        if predicate():
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not client.changed.wait(remaining):
            raise TimeoutError('remote event wait expired')

def worker(rank, results, release):
    client = None
    try:
        client = module.TPEventClient(tuple(cfg['address']), run_id=cfg['run'],
            token=cfg['token'], group='cpu-smoke', rank=rank, size=cfg['size'])
        client.wait_ready()
        for attempt in range(cfg['attempts']):
            key = module.EventKey('cpu-smoke:0', str(attempt))
            client.report('smoke', key, 1)
            wait(client, lambda: client.command('smoke', key) == {'action': 'start'})
            client.report('smoke', key, 2)
            command_id, command = client.next_command('smoke', key)
            assert command == {'action': 'start'}
            client.ack_command('smoke', key, command_id)
            wait(client, lambda: client.receipt('smoke', key) == 9)
            client.report('smoke', key, 3)
        client.flush()
        wait(client, lambda: client.command('smoke', key) == {'action': 'stop'})
        results.put({'rank': rank, 'ok': True})
        # No rank may disconnect until every peer has observed STOP. Closing
        # an established connection deliberately fails its group closed.
        if not release.wait(30):
            raise TimeoutError('remote smoke teardown barrier expired')
    except BaseException:
        results.put({'rank': rank, 'ok': False, 'error': traceback.format_exc()})
    finally:
        if client is not None:
            client.close()

context = multiprocessing.get_context('fork')
results = context.Queue()
release = context.Event()
children = [context.Process(target=worker, args=(rank, results, release))
            for rank in range(1, cfg['size'])]
answers = []
try:
    for child in children:
        child.start()
    for child in children:
        answers.append(results.get(timeout=60))
    release.set()
    for child in children:
        child.join(timeout=5)
    print(json.dumps({'ranks': sorted(answers, key=lambda item: item['rank'])}), flush=True)
finally:
    release.set()
    for child in children:
        if child.pid is not None and child.is_alive():
            child.terminate()
        if child.pid is not None:
            child.join(timeout=3)
if not all(item['ok'] for item in answers) or len(answers) != cfg['size'] - 1:
    raise SystemExit(1)
"""


def wait(client, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        client.changed.clear()
        if predicate():
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not client.changed.wait(remaining):
            raise TimeoutError("rank0 event wait expired")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer", required=True, help="SSH alias of second node")
    parser.add_argument("--listen", required=True, help="local IP reachable by peer")
    parser.add_argument("--peer-python", default="python3")
    parser.add_argument("--tp", type=int, default=8, choices=(2, 8))
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    if args.attempts < 1 or args.attempts > 64:
        parser.error("attempts must be between 1 and 64")
    source = Path(__file__).resolve().parents[1] / (
        "python/sglang/srt/disaggregation/agentic_tp_events.py")
    spec = importlib.util.spec_from_file_location("dualpd_tp_smoke_protocol", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    run = "cpu-tp-events-" + secrets.token_hex(8)
    token = secrets.token_hex(32)
    server = module.TPEventServer(run, token, (args.listen, 0))
    leader = None
    process = None
    started = time.monotonic()
    try:
        leader = module.TPEventClient(server.address, run_id=run, token=token,
                                      group="cpu-smoke", rank=0, size=args.tp)
        leader.wait_ready()
        command = "env CUDA_VISIBLE_DEVICES='' " + shlex.quote(args.peer_python)
        command += " -u -c " + shlex.quote(REMOTE_BOOTSTRAP)
        process = subprocess.Popen(
            ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
             "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
             args.peer, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        process.stdin.write(json.dumps({
            "source": source.read_text(), "worker": REMOTE_WORKER,
            "config": {"address": server.address, "run": run, "token": token,
                       "size": args.tp, "attempts": args.attempts},
        }))
        process.stdin.close()
        process.stdin = None
        for attempt in range(args.attempts):
            key = module.EventKey("cpu-smoke:0", str(attempt))
            leader.report("smoke", key, 1)
            wait(leader, lambda: leader.group_status("smoke", key) == 1)
            leader.publish_command("smoke", key, {"action": "start"})
            leader.report("smoke", key, 2)
            wait(leader, lambda: leader.command("smoke", key) == {"action": "start"})
            command_id, _ = leader.next_command("smoke", key)
            leader.ack_command("smoke", key, command_id)
            wait(leader, lambda: leader.group_status("smoke", key) == 2)
            wait(leader, lambda: leader.command_complete("smoke", key))
            leader.publish_receipt("smoke", key, 9)
            leader.report("smoke", key, 3)
            wait(leader, lambda: leader.group_status("smoke", key) == 3)
        leader.publish_command("smoke", key, {"action": "stop"})
        stdout, stderr = process.communicate(timeout=30)
        if process.returncode:
            raise RuntimeError("remote CPU workers failed: " + stderr + stdout)
        # SSH login banners may precede the machine-readable record.
        remote = json.loads(stdout.strip().splitlines()[-1])
        if not all(item["ok"] for item in remote["ranks"]):
            raise RuntimeError("remote worker did not complete")
        print(json.dumps({
            "test": "two-node CPU control messages only", "ok": True,
            "local_host": socket.gethostname(), "peer": args.peer,
            "tp": args.tp, "attempts": args.attempts,
            "seconds": round(time.monotonic() - started, 3),
            "gpu_test": False, "engine_integration_test": False,
            "remote": remote,
        }, indent=2))
    finally:
        if leader is not None:
            leader.close()
        server.close()
        if process is not None and process.poll() is None:
            # Closing control sockets wakes the CPU workers. Allow their own
            # finally blocks to reap children before terminating only this SSH.
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.communicate(timeout=10)


if __name__ == "__main__":
    main()
