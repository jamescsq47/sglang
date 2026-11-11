"""Public, lazy application-ACK adapter for the node-local DualPD protocol.

Applications send request/generation notifications, never manipulate internal
stores or ownership records themselves. This adapter preserves the existing
marker protocol; global/multi-node terminality continues to use the Router's
HTTP application_final endpoint instead of shared filesystem state.
"""
from __future__ import annotations

import os
from typing import Any


def publish_local_ack(
    p_ready_dir: str, request_id: str, generation: int, *, event: str
) -> dict[str, Any]:
    """Publish a tool/final marker without allocating or releasing KV.

    Keep the stable public module identities: extension sources are under
    dualpd/, but state classes must not be re-imported through that path.
    Errors propagate to the caller's existing ACK failure policy.
    """
    if event not in {"tool", "final"}:
        raise ValueError("application ACK event must be tool or final")
    if not p_ready_dir:
        raise ValueError("node-local application ACK requires p_ready_dir")
    from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
    from sglang.srt.disaggregation.agentic_kv_lifecycle import RequestGeneration

    store = AgenticEarlyClaimStore(os.path.join(p_ready_dir, "early-claims"))
    request = RequestGeneration(str(request_id), generation)
    if event == "final":
        return store.publish_final(request)
    return store.publish_tool(request)
