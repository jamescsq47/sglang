from __future__ import annotations

import copy
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
import torch

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    TransferBackend,
    get_kv_class,
    is_mla_backend,
)

logger = logging.getLogger(__name__)


@dataclass
class AgenticDirectRuntime:
    """A second, role-reversed NIXL data plane for D->P GPU KV transfer."""

    manager: Any
    aux_buffer: torch.Tensor
    transfer_backend: TransferBackend
    bootstrap_server: Optional[Any] = None
    bootstrap_addr: Optional[str] = None
    kv_pool: Any = None

    @property
    def sender_class(self):
        return get_kv_class(self.transfer_backend, KVClassType.SENDER)

    @property
    def receiver_class(self):
        return get_kv_class(self.transfer_backend, KVClassType.RECEIVER)

    @property
    def layout_hash(self) -> str:
        """Stable, address-independent fingerprint of one TP KV shard."""

        args = self.manager.kv_args
        fields = (
            int(getattr(args, "page_size", 1)),
            int(getattr(args, "kv_head_num", 1)),
            int(getattr(args, "total_kv_head_num", 1)),
            tuple(int(value) for value in getattr(args, "kv_item_lens", ())),
            len(getattr(args, "kv_data_ptrs", ())),
        )
        return hashlib.sha256(repr(fields).encode("ascii")).hexdigest()[:24]


def debug_kv_digest(kv_pool, token_indices) -> str | None:
    """Return an exact KV digest when the opt-in diagnostic is enabled."""

    if os.getenv("SGLANG_AGENTIC_KV_DEBUG_DIGEST", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    indices = torch.as_tensor(
        token_indices, dtype=torch.long, device=kv_pool.k_buffer[0].device
    )
    limit = int(os.getenv("SGLANG_AGENTIC_KV_DEBUG_DIGEST_TOKENS", "0"))
    if limit > 0:
        indices = indices[:limit]
    digest = hashlib.sha256()
    for tensor in kv_pool.k_buffer + kv_pool.v_buffer:
        data = tensor.index_select(0, indices).contiguous().view(torch.uint8)
        digest.update(data.cpu().numpy().tobytes())
    return digest.hexdigest()


def _make_kv_args(
    *,
    transfer_backend: TransferBackend,
    kv_pool,
    server_args,
    engine_rank: int,
    pp_rank: int,
    gpu_id: int,
    total_kv_heads: int,
):
    kv_args_class = get_kv_class(transfer_backend, KVClassType.KVARGS)
    kv_args = kv_args_class()
    kv_args.engine_rank = engine_rank
    kv_args.pp_rank = pp_rank
    kv_args.system_dp_rank = 0
    kv_args.prefill_start_layer = kv_pool.start_layer
    (
        kv_args.kv_data_ptrs,
        kv_args.kv_data_lens,
        kv_args.kv_item_lens,
    ) = kv_pool.get_contiguous_buf_infos()
    if not is_mla_backend(kv_pool):
        kv_args.kv_head_num = kv_pool.head_num
        kv_args.total_kv_head_num = total_kv_heads
    kv_args.page_size = kv_pool.page_size

    # NIXL's wire protocol always sends one auxiliary item on the final KV
    # chunk.  The reverse path does not need SGLang request metadata, but a
    # registered one-byte DRAM item keeps the stock sender/receiver protocol
    # intact instead of forking it.
    aux_buffer = torch.zeros(1, dtype=torch.uint8, pin_memory=True)
    kv_args.aux_data_ptrs = [aux_buffer.data_ptr()]
    kv_args.aux_data_lens = [aux_buffer.nbytes]
    kv_args.aux_item_lens = [aux_buffer.nbytes]
    kv_args.state_data_ptrs = []
    kv_args.state_data_lens = []
    kv_args.state_item_lens = []
    kv_args.state_dim_per_tensor = []
    kv_args.state_type = "none"
    kv_args.ib_device = server_args.disaggregation_ib_device
    kv_args.ib_traffic_class = getattr(
        server_args, "disaggregation_ib_traffic_class", ""
    )
    kv_args.gpu_id = gpu_id
    return kv_args, aux_buffer


def create_agentic_direct_runtime(
    *,
    role: DisaggregationMode,
    kv_pool,
    server_args,
    engine_rank: int,
    pp_rank: int,
    gpu_id: int,
    total_kv_heads: int,
    bootstrap_port: Optional[int] = None,
) -> AgenticDirectRuntime:
    """Create an isolated reverse manager without mutating the normal PD plane.

    D takes the PREFILL/sender role and owns a dedicated bootstrap port; P
    takes the DECODE/receiver role.  Only NIXL is accepted because this fast
    path is specifically GPU-to-GPU and must not silently become a storage Put.
    """

    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)
    if transfer_backend is not TransferBackend.NIXL:
        raise ValueError(
            "Agentic direct D->P transfer currently requires the NIXL PD backend"
        )
    if role is DisaggregationMode.PREFILL and not bootstrap_port:
        raise ValueError("reverse sender requires a dedicated bootstrap port")

    direct_args = copy.copy(server_args)
    bootstrap_server = None
    tp_size = max(1, int(getattr(server_args, "tp_size", 1)))
    tp_rank = int(engine_rank) % tp_size
    if role is DisaggregationMode.PREFILL:
        direct_args.disaggregation_bootstrap_port = int(bootstrap_port)
        # One logical TP engine owns one bootstrap rank table.  Creating a
        # server in every scheduler process makes rank 1 collide with rank 0's
        # TCP port; only rank 0 listens while every rank registers its own KV
        # shard in that common table.
        if tp_rank == 0:
            bootstrap_class = get_kv_class(
                transfer_backend, KVClassType.BOOTSTRAP_SERVER
            )
            bootstrap_server = bootstrap_class(
                host=direct_args.host,
                port=direct_args.disaggregation_bootstrap_port,
            )
        health_url = (
            f"http://127.0.0.1:{direct_args.disaggregation_bootstrap_port}/health"
        )
        # TP scheduler ranks initialize independently.  In particular, rank 0
        # may still be constructing HiCache/Mooncake state while a follower
        # is already ready to register.  A five-second deadline made startup
        # nondeterministic for large models even though the listener came up
        # immediately afterwards.
        bootstrap_timeout = float(
            os.getenv("SGLANG_AGENTIC_KV_DIRECT_BOOTSTRAP_TIMEOUT_SECONDS", "60")
        )
        deadline = time.monotonic() + max(5.0, bootstrap_timeout)
        while time.monotonic() < deadline:
            try:
                if requests.get(health_url, timeout=0.2).status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.02)
        else:
            raise RuntimeError(
                f"reverse bootstrap server did not start on {health_url}"
            )

    kv_args, aux_buffer = _make_kv_args(
        transfer_backend=transfer_backend,
        kv_pool=kv_pool,
        server_args=direct_args,
        engine_rank=engine_rank,
        pp_rank=pp_rank,
        gpu_id=gpu_id,
        total_kv_heads=total_kv_heads,
    )
    manager_class = get_kv_class(transfer_backend, KVClassType.MANAGER)
    manager = manager_class(
        kv_args,
        role,
        direct_args,
        is_mla_backend(kv_pool),
    )
    bootstrap_addr = None
    if role is DisaggregationMode.PREFILL:
        # Registration is HTTP and all TP scheduler processes start
        # independently.  Rank 0 can inspect the in-process table; followers
        # repeatedly register until the common endpoint is reachable and then
        # return without trying to own the listener.
        deadline = time.monotonic() + 10.0
        registered = False
        while time.monotonic() < deadline:
            try:
                manager.register_to_bootstrap()
                registered = True
            except Exception:
                time.sleep(0.02)
                continue
            if bootstrap_server is None or bootstrap_server._is_ready():
                break
            time.sleep(0.02)
        if not registered or (
            bootstrap_server is not None and not bootstrap_server._is_ready()
        ):
            raise RuntimeError(
                "reverse NIXL TP rank registration did not complete "
                f"rank={tp_rank}/{tp_size}"
            )
        bootstrap_addr = f"{manager.local_ip}:{direct_args.disaggregation_bootstrap_port}"

    return AgenticDirectRuntime(
        manager=manager,
        aux_buffer=aux_buffer,
        transfer_backend=transfer_backend,
        bootstrap_server=bootstrap_server,
        bootstrap_addr=bootstrap_addr,
        kv_pool=kv_pool,
    )
