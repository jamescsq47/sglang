from __future__ import annotations

import os


def rank_env_int(
    scalar_name: str,
    vector_name: str,
    *,
    tp_rank: int,
    default: int = -1,
) -> int:
    """Read a rank-local integer while preserving the TP=1 scalar API."""

    raw_vector = os.getenv(vector_name, "").strip()
    if raw_vector:
        values = [part.strip() for part in raw_vector.split(",")]
        if not 0 <= int(tp_rank) < len(values):
            raise ValueError(
                f"{vector_name} has {len(values)} entries, no TP rank {tp_rank}"
            )
        return int(values[int(tp_rank)])
    raw_scalar = os.getenv(scalar_name)
    return int(raw_scalar) if raw_scalar not in {None, ""} else int(default)


def rank_scoped_arena_directory(
    directory: str,
    *,
    tp_rank: int,
    tp_size: int,
    numa_node: int,
) -> str:
    """Give each TP shard an independent, NUMA-local physical arena."""

    if int(tp_size) == 1:
        return directory
    return os.path.join(
        directory,
        f"tp-rank-{int(tp_rank)}-numa-{int(numa_node)}",
    )


def logical_tp_claim_id(prefix: str, engine_id: str, snapshot_id: str) -> str:
    return f"{prefix}:{engine_id}:{snapshot_id}"


def request_generation_key(rid: str, bootstrap_room: int | str) -> str:
    """Stable identity carried by rank-0 commands for one wire generation."""

    return f"{rid}@{bootstrap_room}"
