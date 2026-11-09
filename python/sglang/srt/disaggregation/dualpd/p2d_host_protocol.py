"""Pure P-to-D wire declarations; no Host state, worker or device imports."""
from __future__ import annotations

P2D_CUSTOM_SNAPSHOT_ID = "agentic_p2d_host_snapshot_id"
P2D_CUSTOM_PREFILL_DOMAIN = "agentic_p2d_prefill_domain"


def p2d_snapshot_id(bootstrap_room: int) -> str:
    """Return a collision-free id in the direction-specific ledger."""

    return f"p2d:{int(bootstrap_room)}"
