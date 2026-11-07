"""Isolated pre-consolidation router paired with the pd_mamba ledger API.

The current pd router uses different recovery-CAS fields; renaming one method
would not preserve snapshot ownership. Load the complete matching immutable
implementation instead, without editing the shared router or either environment.
"""
from __future__ import annotations

import hashlib
import json
import linecache
import logging
import os
from pathlib import Path
import sys
import types
import math

REVISION = "82e18c19e21a85e0ef3dc1a52784d2fa4f788173"
SOURCE_PATH = "examples/pd/late_binding_router.py"
PINNED_SOURCE = Path(__file__).with_name("legacy_mamba_pinned.py")


def load_router_module():
    # Vendored byte-identical revision: no runtime dependency on slime or Git.
    source = PINNED_SOURCE.read_text()
    filename = f"<pd_mamba_router:{REVISION}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    module = types.ModuleType("pd_mamba_pinned_router")
    module.__file__ = filename
    sys.modules[module.__name__] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    original_sync_get = module._sync_json_get

    def compatible_sync_get(url, timeout):
        result = original_sync_get(url, timeout)
        if url.endswith("/get_load"):
            rows = [result] if isinstance(result, dict) else result
            for row in rows:
                if "num_physical_used_tokens" not in row:
                    total = int(row.get("num_tokens", 0))
                    pending = int(row.get("num_pending_tokens", 0))
                    if total < 0 or pending < 0 or pending > total:
                        raise RuntimeError(f"Invalid legacy token accounting at {url}")
                    row["num_physical_used_tokens"] = total - pending
        return result

    # Only this private module is adapted; keep D's existing capacity lookup,
    # reservation/CAS logic and HTTP offload thread unchanged.
    module._sync_json_get = compatible_sync_get
    return module, hashlib.sha256(source.encode()).hexdigest()


def router_class(module):
    class MambaTP1Router(module.LateBindingMiniLoadBalancer):
        async def _fetch_prefill_hbm_pressure(self, session, url):
            # SGLang 0.5.18 projects /get_load onto legacy core fields and
            # omits the physical pool capacity. Never interpret missing as 0.
            timeout = module.aiohttp.ClientTimeout(total=self.load_timeout)
            async with session.get(f"{url}/get_load", timeout=timeout) as response:
                response.raise_for_status()
                rows = await response.json()
            if isinstance(rows, dict):
                rows = [rows]
            if not rows:
                raise RuntimeError(f"Empty Prefill load response: {url}")
            if any("max_total_num_tokens" not in row for row in rows):
                if int(os.environ.get("SGLANG_AGENTIC_KV_TP_SIZE", "1")) != 1 or len(rows) != 1:
                    raise RuntimeError("Capacity fallback requires one TP1/DP1 worker")
                capacities = getattr(self, "_mamba_tp1_capacities", None)
                if capacities is None:
                    capacities = self._mamba_tp1_capacities = {}
                if url not in capacities:
                    async with session.get(f"{url}/server_info", timeout=timeout) as response:
                        response.raise_for_status()
                        info = await response.json()
                    value = info.get("max_total_num_tokens")
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or int(value) != value:
                        raise RuntimeError(f"Invalid actual Prefill KV capacity at {url}: {value!r}")
                    if int(info.get("tp_size", 1)) != 1 or int(info.get("dp_size", 1)) != 1:
                        raise RuntimeError("Capacity fallback does not aggregate TP/DP replicas")
                    capacities[url] = int(value)
                translated = []
                for row in rows:
                    total = int(row.get("num_tokens", 0))
                    pending = int(row.get("num_pending_tokens", 0))
                    if total < 0 or pending < 0 or pending > total:
                        raise RuntimeError(f"Invalid legacy token accounting at {url}")
                    translated.append({
                        **row,
                        "max_total_num_tokens": capacities[url],
                        # Legacy total includes queued, not-yet-allocated
                        # prompts. Router shadow work is accounted separately.
                        "num_physical_used_tokens": row.get("num_physical_used_tokens", total - pending),
                    })
                rows = translated
            used = [int(row.get("num_physical_used_tokens", row.get("num_tokens", 0))) for row in rows]
            capacities = [int(row["max_total_num_tokens"]) for row in rows]
            free = [max(0, capacity - occupied) for capacity, occupied in zip(capacities, used)]
            capacity = min(capacities)
            waiting = sum(int(row.get("num_waiting_reqs", 0)) for row in rows)
            return max(0, capacity - min(free)), capacity, waiting, free

    return MambaTP1Router


def main():
    import sglang_router.mini_lb as mini_lb_module
    from sglang_router.launch_router import parse_router_args
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    module, digest = load_router_module()
    run_dir = os.environ.get("RUN_DIR")
    if run_dir:
        (Path(run_dir) / "router_source.json").write_text(json.dumps({
            "revision": REVISION, "path": SOURCE_PATH, "sha256": digest,
            "reason": "Pair router recovery assignment/CAS with isolated pd_mamba ledger API",
        }, indent=2) + "\n")
    args = parse_router_args(sys.argv[1:])
    args.mini_lb = True
    router = router_class(module)(args)
    mini_lb_module.lb = router
    mini_lb_module.app.router.add_event_handler("shutdown", router.close)
    if router.enable_trace:
        mini_lb_module.process_tracing_init(router.otlp_traces_endpoint, "sglang")
        mini_lb_module.trace_set_thread_info("Mini lb")
    uvicorn.run(mini_lb_module.app, host=router.host, port=router.port, loop="asyncio")


if __name__ == "__main__":
    main()
