import asyncio
import base64
import random
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
import orjson
import sglang.srt.disaggregation.dualpd.router.late_binding_router as late_binding_router_module
from sglang.srt.disaggregation.agentic_early_claim import AgenticEarlyClaimStore
from sglang.srt.disaggregation.agentic_kv_lifecycle import (
    AgenticRequestMetadata,
    RequestGeneration,
)
from sglang.srt.disaggregation.agentic_host_staging import (
    HostStageState,
    SharedHostStagingLedger,
)
from sglang.srt.disaggregation.p2d_host_staging import (
    P2D_CUSTOM_PREFILL_DOMAIN,
    P2D_CUSTOM_SNAPSHOT_ID,
    p2d_snapshot_id,
)
from sglang.srt.disaggregation.agentic_prefill_pressure import (
    SharedPrefillPressureReservations,
)

from sglang.srt.disaggregation.dualpd.router.late_binding_router import (
    DecodeLoad,
    DecodeReservation,
    LateBindingMiniLoadBalancer,
    _PReadyAdmission,
    _PrefillAdmissionGate,
    _PrefillAdmissionWaiter,
    _PrefillWorkReservation,
)


class LateBindingRouterTest(unittest.IsolatedAsyncioTestCase):
    def make_router(self, ready_dir: Path) -> LateBindingMiniLoadBalancer:
        router = object.__new__(LateBindingMiniLoadBalancer)
        router.p_ready_dir = ready_dir
        router.ready_timeout = 1.0
        router.prefill_accept_timeout = 1.0
        router.prefill_queue_timeout = 1.0
        router.ready_poll_interval = 0.001
        router.load_timeout = 1.0
        router.timeout = 10.0
        router.reservation_timeout = 1.0
        router.decode_headroom_tokens = 512
        router.max_decode_admission_tokens = 4096
        router.request_load_weight = 0.05
        router.transfer_request_weight = 2.0
        router.max_direct_transfer_inflight = 0
        router.context_token_floor = 2048
        router.context_token_ceiling = 8192
        router.wait_for_feasible_decode = True
        router.target_decode_kv_fraction = 1.0
        router.no_capacity_poll_interval = 0.001
        router.load_cache_ttl = 0.05
        router.decode_urls = ["http://d0", "http://d1"]
        router._selection_lock = asyncio.Lock()
        router._p_ready_submitted_sequences = set()
        router._p_ready_fifo_waiters = {}
        router._p_ready_fifo_events = {}
        router._p_ready_fifo_dispatchers = {}
        router._p_ready_fifo_active = {}
        router._p_ready_monitor_task = None
        router._p_ready_waiters = {}
        router._p_ready_snapshot = {}
        router._reservations = {}
        router._last_loads = {}
        router._load_cache = []
        router._load_cache_at = 0.0
        router._load_cache_sample_started_at = 0.0
        router._load_sample_started_at_by_url = {}
        router._load_refresh_task = None
        router._backend_session = None
        router._load_session = None
        router._admitted_reservation_at = {}
        router._legacy_load_urls = set()
        router.prefill_urls = ["http://p0", "http://p1"]
        router._prefill_work_lock = asyncio.Lock()
        router._prefill_pending_tokens = [0, 0]
        router._prefill_pending_requests = [0, 0]
        router._prefill_direct_pending_tokens = [0, 0]
        router._prefill_work_tiebreak = 0
        router.ablation_random_routing = False
        router.p2d_prebind_ablation = False
        router._routing_rng = random.Random(2026)
        router._prefill_pressure_domains = []
        router._prefill_pressure_at = 0.0
        router._prefill_pressure_sample_started_at = 0.0
        router._prefill_pressure_interval = 0.2
        async def close_sessions():
            for name in ("_load_session", "_backend_session"):
                session = getattr(router, name, None)
                if session is not None and not session.closed:
                    await session.close()

        self.addAsyncCleanup(close_sessions)
        return router

    async def test_p2d_prebind_submits_decode_before_prefill_and_skips_host(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.p2d_prebind_ablation = True
            router.p2d_host_staging = False
            router.numa_domains = False
            router.dynamic_prefill_domains = False
            router.early_claim_store = None
            router.max_prefill_inflight = 4
            router._prefill_admission = _PrefillAdmissionGate(4)
            router._wait_until_prefill_accepted = AsyncMock()
            router._wait_until_prefill_scheduled = AsyncMock()
            router._wait_until_prefill_ready = AsyncMock(return_value=1024)
            router._stage_p2d_until_durable = AsyncMock()
            reservation = DecodeReservation(
                reservation_id="prebound",
                url="http://d0",
                prompt_tokens=1024,
                admission_tokens=1536,
                request_count=1,
                rooms=(7,),
                created_at=0.0,
            )
            router._select_and_reserve_decode = AsyncMock(
                return_value=reservation
            )
            router._release_reservation_when_admitted = AsyncMock()
            decode_submitted = asyncio.Event()

            class Response:
                status = 200

                async def read(self):
                    return b"{}"

                def release(self):
                    return None

            class Session:
                def __init__(self):
                    self.calls = []
                    self.prefill_stream = None

                async def post(self, url, **kwargs):
                    self.calls.append(url)
                    if url == "http://d0/generate":
                        decode_submitted.set()
                    elif url == "http://p0/generate":
                        self.assert_decode_was_first = decode_submitted.is_set()
                        self.prefill_stream = kwargs["json"].get("stream")
                    return Response()

            session = Session()
            await router._late_dispatch(
                session,
                {
                    "bootstrap_room": 7,
                    "input_ids": list(range(1024)),
                    "sampling_params": {},
                },
                "http://p0",
                "generate",
                {},
            )

            self.assertTrue(session.assert_decode_was_first)
            self.assertEqual(
                session.calls,
                ["http://d0/generate", "http://p0/generate"],
            )
            self.assertIs(session.prefill_stream, False)
            router._stage_p2d_until_durable.assert_not_awaited()
            router._select_and_reserve_decode.assert_awaited_once()
            await router.close()

    async def test_p2d_prebind_selection_failure_does_not_take_p_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.p2d_prebind_ablation = True
            router.p2d_host_staging = False
            router.numa_domains = False
            router.dynamic_prefill_domains = False
            router.early_claim_store = None
            router.max_prefill_inflight = 1
            router._prefill_admission = _PrefillAdmissionGate(1)
            router._select_and_reserve_decode = AsyncMock(
                side_effect=TimeoutError("no D capacity")
            )

            class Session:
                async def post(self, *_args, **_kwargs):
                    raise AssertionError("no backend request should be submitted")

            with self.assertRaisesRegex(TimeoutError, "no D capacity"):
                await router._late_dispatch(
                    Session(),
                    {
                        "bootstrap_room": 8,
                        "input_ids": [1, 2, 3],
                        "sampling_params": {},
                    },
                    "http://p0",
                    "generate",
                    {},
                )
            self.assertEqual(router._prefill_admission.active, 0)

    async def test_p2d_prebind_wait_has_deadline_after_decode_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.ready_timeout = 0.01

            class Response:
                status = 200

            async def pending():
                await asyncio.Future()

            prefill_task = asyncio.create_task(pending())
            decode_task = asyncio.create_task(asyncio.sleep(0, result=Response()))
            try:
                with self.assertRaisesRegex(
                    TimeoutError, "native pre-bound Prefill"
                ):
                    await router._wait_prebound_prefill_or_redirect(
                        prefill_task, decode_task, None
                    )
            finally:
                prefill_task.cancel()
                await asyncio.gather(prefill_task, return_exceptions=True)

    async def test_p2d_prebind_keeps_d_credit_when_abort_is_not_acknowledged(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.p2d_prebind_ablation = True
            router.p2d_host_staging = False
            router.numa_domains = False
            router.dynamic_prefill_domains = False
            router.early_claim_store = None
            router.max_prefill_inflight = 1
            router._prefill_admission = _PrefillAdmissionGate(1)
            reservation = DecodeReservation(
                reservation_id="abort-not-acked",
                url="http://d0",
                prompt_tokens=3,
                admission_tokens=515,
                request_count=1,
                rooms=(9,),
                created_at=time.monotonic(),
            )

            async def reserve(*_args, **_kwargs):
                router._reservations[reservation.reservation_id] = reservation
                return reservation

            router._select_and_reserve_decode = reserve
            router._wait_until_prefill_accepted = AsyncMock(
                side_effect=RuntimeError("P failed")
            )
            async def abort_prefill(_session, _server, _request, task):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                return True

            router._abort_prefill_attempt = AsyncMock(side_effect=abort_prefill)
            router._abort_decode_attempt = AsyncMock(return_value=False)

            class Session:
                async def post(self, *_args, **_kwargs):
                    await asyncio.Future()

            with self.assertRaisesRegex(RuntimeError, "P failed"):
                await router._late_dispatch(
                    Session(),
                    {
                        "bootstrap_room": 9,
                        "input_ids": [1, 2, 3],
                        "sampling_params": {},
                    },
                    "http://p0",
                    "generate",
                    {},
                )
            self.assertEqual(router._prefill_admission.active, 0)
            self.assertIn("abort-not-acked", router._reservations)

    async def test_p2d_prebind_cancel_after_p_fence_aborts_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.p2d_prebind_ablation = True
            router.p2d_host_staging = False
            router.numa_domains = False
            router.dynamic_prefill_domains = False
            router.early_claim_store = None
            router.max_prefill_inflight = 1
            router._prefill_admission = _PrefillAdmissionGate(1)
            router._wait_until_prefill_accepted = AsyncMock()
            reservation = DecodeReservation(
                reservation_id="cancel-after-p",
                url="http://d0",
                prompt_tokens=3,
                admission_tokens=515,
                request_count=1,
                rooms=(10,),
                created_at=time.monotonic(),
            )

            async def reserve(*_args, **_kwargs):
                router._reservations[reservation.reservation_id] = reservation
                return reservation

            router._select_and_reserve_decode = reserve
            router._abort_prefill_attempt = AsyncMock(return_value=True)
            router._abort_decode_attempt = AsyncMock(return_value=True)
            p_fenced = asyncio.Event()

            class Response:
                status = 200

                async def read(self):
                    p_fenced.set()
                    return b"{}"

                def release(self):
                    return None

            class Session:
                async def post(self, url, **_kwargs):
                    if url == "http://d0/generate":
                        await asyncio.Future()
                    return Response()

            task = asyncio.create_task(
                router._late_dispatch(
                    Session(),
                    {
                        "bootstrap_room": 10,
                        "input_ids": [1, 2, 3],
                        "sampling_params": {},
                    },
                    "http://p0",
                    "generate",
                    {},
                )
            )
            await asyncio.wait_for(p_fenced.wait(), timeout=1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

            router._abort_decode_attempt.assert_awaited_once()
            self.assertEqual(router._prefill_admission.active, 0)
            self.assertEqual(router._active_prefill_attempts, {})

    async def test_p2d_prebind_redirect_replaces_owner_without_leaking_old_room(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.p2d_prebind_ablation = True
            router.p2d_host_staging = False
            router.numa_domains = False
            router.dynamic_prefill_domains = True
            router.early_claim_store = None
            router.max_prefill_inflight = 1
            router._prefill_admission = _PrefillAdmissionGate(1)
            router._prefill_admissions = [
                _PrefillAdmissionGate(1),
                _PrefillAdmissionGate(1),
            ]
            router._wait_until_prefill_accepted = AsyncMock()
            router._abort_prefill_attempt = AsyncMock(return_value=True)
            router._abort_decode_attempt = AsyncMock(return_value=True)
            router._ensure_prefill_pressure_monitor = lambda: None
            work = _PrefillWorkReservation(domain=0, tokens=3)
            router._prefill_pending_tokens = [3, 0]
            router._prefill_pending_requests = [1, 0]
            router._resolve_dynamic_prefill_work = AsyncMock(return_value=work)
            reservations = [
                DecodeReservation(
                    reservation_id=f"redirect-{index}",
                    url="http://d0",
                    prompt_tokens=3,
                    admission_tokens=515,
                    request_count=1,
                    rooms=(10 + index,),
                    created_at=time.monotonic(),
                )
                for index in range(2)
            ]

            async def reserve(*_args, **_kwargs):
                reservation = reservations.pop(0)
                router._reservations[reservation.reservation_id] = reservation
                return reservation

            router._select_and_reserve_decode = reserve

            class Response:
                status = 200

                async def read(self):
                    return b"{}"

                def release(self):
                    return None

            router._wait_prebound_prefill_or_redirect = AsyncMock(
                side_effect=[
                    late_binding_router_module._PrefillRedirect(1, "host"),
                    Response(),
                ]
            )

            class Session:
                async def post(self, *_args, **_kwargs):
                    return Response()

            request = {
                "bootstrap_room": 10,
                "input_ids": [1, 2, 3],
                "sampling_params": {},
            }
            await router._late_dispatch(
                Session(), request, "http://p0", "generate", {}
            )

            self.assertNotIn(10, router._active_prefill_attempts)
            self.assertEqual(router._active_prefill_attempts, {})

    def test_p2d_host_metadata_uses_chat_custom_params(self):
        request = {
            "custom_params": {"agentic_request_id": "chat-request"},
            "messages": [{"role": "user", "content": "hello"}],
        }

        LateBindingMiniLoadBalancer._set_p2d_host_metadata(
            request, "p2d:41", 1
        )

        self.assertNotIn("sampling_params", request)
        self.assertEqual(
            request["custom_params"],
            {
                "agentic_request_id": "chat-request",
                P2D_CUSTOM_SNAPSHOT_ID: "p2d:41",
                P2D_CUSTOM_PREFILL_DOMAIN: 1,
            },
        )

    def test_p2d_host_metadata_uses_generate_sampling_params(self):
        request = {
            "sampling_params": {
                "custom_params": {"agentic_request_id": "generate-request"}
            }
        }

        LateBindingMiniLoadBalancer._set_p2d_host_metadata(
            request, "p2d:42", 0
        )

        self.assertNotIn("custom_params", request)
        self.assertEqual(
            request["sampling_params"]["custom_params"],
            {
                "agentic_request_id": "generate-request",
                P2D_CUSTOM_SNAPSHOT_ID: "p2d:42",
                P2D_CUSTOM_PREFILL_DOMAIN: 0,
            },
        )

    async def test_random_routing_ablation_ignores_load_but_keeps_feasibility(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router.ablation_random_routing = True
        router._routing_rng = random.Random(2026)
        router._prefill_pending_tokens = [0, 1_000_000]

        selected = [
            (await router._reserve_prefill_work(1)).domain for _ in range(4)
        ]

        # The seeded random policy selects the heavily loaded P1 as well;
        # load-aware routing would select only P0 for this setup.
        self.assertEqual(selected, [0, 1, 0, 0])

        loads = [
            (False, 1.0, 0.1, 1, types.SimpleNamespace(url="http://d0")),
            (False, 10_000.0, 0.8, 100, types.SimpleNamespace(url="http://d1")),
        ]
        router._routing_rng = random.Random(2026)
        chosen = [router._choose_decode_score(loads)[4].url for _ in range(4)]
        self.assertEqual(
            chosen,
            ["http://d0", "http://d1", "http://d0", "http://d0"],
        )

    async def test_prefill_selection_uses_only_remaining_kv(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router._prefill_pressure_at = time.monotonic()
        router._prefill_pressure_domains = [
            {
                "domain": 0,
                "hbm_capacity_tokens": 100_000,
                "hbm_used_tokens": 10_000,
                "arena_capacity_bytes": 100,
                "arena_used_bytes": 0,
                "p2d_arena_capacity_bytes": 100,
                "p2d_inflight_tokens": 80_000,
                "p2d_inflight_requests": 40,
                "p2d_host_tokens": 20_000,
                "p2d_host_requests": 10,
                "p2d_host_bytes": 80,
            },
            {
                "domain": 1,
                "hbm_capacity_tokens": 100_000,
                "hbm_used_tokens": 20_000,
                "arena_capacity_bytes": 100,
                "arena_used_bytes": 0,
                "p2d_arena_capacity_bytes": 100,
                "p2d_inflight_tokens": 0,
                "p2d_inflight_requests": 0,
                "p2d_host_tokens": 0,
                "p2d_host_requests": 0,
                "p2d_host_bytes": 0,
            },
        ]

        reservation = await router._reserve_prefill_work(1000)
        # P0 has more physical KV remaining.  Queue and Host-pressure fields
        # are intentionally ignored by the simplified D->P policy.
        self.assertEqual(reservation.domain, 0)

    async def test_slow_h2d_retargets_to_largest_remaining_kv(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router._prefill_pressure_at = time.monotonic()
        router._prefill_pressure_domains = [
            {
                "domain": 0,
                "hbm_capacity_tokens": 100_000,
                "hbm_used_tokens": 10_000,
                "arena_capacity_bytes": 100,
                "arena_used_bytes": 95,
            },
            {
                "domain": 1,
                "hbm_capacity_tokens": 100_000,
                "hbm_used_tokens": 80_000,
                "arena_capacity_bytes": 100,
                "arena_used_bytes": 0,
            },
        ]
        reservation = _PrefillWorkReservation(
            domain=1, tokens=500, direct_workset_tokens=10_000
        )
        router._prefill_pending_tokens[1] = 500
        router._prefill_pending_requests[1] = 1
        router._prefill_direct_pending_tokens[1] = 10_000

        selected = await router._move_prefill_work_to_max_remaining_kv(
            reservation, 10_000
        )

        self.assertEqual(selected, 0)
        self.assertEqual(reservation.domain, 0)
        self.assertEqual(router._prefill_direct_pending_tokens, [10_000, 0])

    async def test_slow_recovery_assignment_loser_waits_for_recompute_state(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        parent = RequestGeneration("eviction-assignment-race", 0)
        states = iter(
            [HostStageState.EVICTING.value, HostStageState.RECOMPUTE_REQUIRED.value]
        )
        published = []
        removed = []
        router._d2p_host_ledger = types.SimpleNamespace(
            assign_d2p_recovery_domain=lambda *_args: False,
            get=lambda *_args: {"state": next(states), "arena_domain": 0},
        )
        router.early_claim_store = types.SimpleNamespace(
            publish_route=lambda *args, **kwargs: published.append((args, kwargs)),
            remove_arrival=lambda request: removed.append(request),
        )

        selected = await router._commit_slow_recovery_domain(
            parent,
            {"route": "host_ready", "prefill_domain": 0},
            1,
        )

        self.assertIsNone(selected)
        self.assertEqual(published[0][1]["route"], "recompute")
        self.assertEqual(removed, [parent])

    async def test_watcher_eviction_loser_restores_original_p_accounting(self):
        router = self.make_router(Path("/dev/shm/test-eviction-accounting"))
        parent = RequestGeneration("eviction-accounting", 0)
        request = {"input_ids": list(range(9000))}
        route = {
            "route": "host_ready",
            "prefill_domain": 0,
            "snapshot_tokens": 8000,
        }
        router.early_claim_store = types.SimpleNamespace(
            read_route=lambda *_args, **_kwargs: route,
            read_final=lambda *_args, **_kwargs: None,
        )
        router._d2p_host_ledger = types.SimpleNamespace(
            get=lambda *_args: {"state": HostStageState.HOST_READY.value}
        )
        router._prefill_pressure_at = time.monotonic()
        router._prefill_pressure_domains = [
            {"domain": 0, "hbm_capacity_tokens": 20_000, "hbm_used_tokens": 15_000},
            {"domain": 1, "hbm_capacity_tokens": 20_000, "hbm_used_tokens": 0},
        ]
        reservation = _PrefillWorkReservation(
            domain=0, tokens=1000, direct_workset_tokens=9000
        )
        router._prefill_pending_tokens = [1000, 0]
        router._prefill_pending_requests = [1, 0]
        router._prefill_direct_pending_tokens = [9000, 0]
        router._commit_slow_recovery_domain = AsyncMock(return_value=None)
        metadata = AgenticRequestMetadata(
            request_id="eviction-accounting",
            generation=1,
            parent_generation=0,
        )

        outcome = await router._watch_dynamic_prefill_route(
            request, metadata, reservation
        )

        self.assertEqual(outcome, {"action": "recompute", "route": "host_evicted"})
        self.assertEqual(reservation.domain, 0)
        self.assertEqual(router._prefill_pending_tokens, [9000, 0])
        self.assertEqual(router._prefill_direct_pending_tokens, [0, 0])

    def test_d2p_host_remaining_counts_all_extent_owning_states(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        entries = {}
        live_states = (
            HostStageState.HOST_RESERVED,
            HostStageState.HOST_WRITING,
            HostStageState.HOST_READY,
            HostStageState.H2D_LOADING,
            HostStageState.HBM_READY,
            HostStageState.RETRY_PENDING,
            HostStageState.ABORTING,
            HostStageState.EVICTING,
        )
        for index, state in enumerate(live_states):
            entries[f"live-{index}"] = {
                "state": state.value,
                "arena_domain": index % 2,
                "byte_size": index + 1,
            }
        entries["released"] = {
            "state": HostStageState.CONSUMED.value,
            "arena_domain": 0,
            "byte_size": 10_000,
        }
        entries["recompute"] = {
            "state": HostStageState.RECOMPUTE_REQUIRED.value,
            "arena_domain": 1,
            "byte_size": 10_000,
        }
        router._d2p_host_ledger = types.SimpleNamespace(
            snapshot_entries=lambda: entries
        )

        self.assertEqual(router._prefill_arena_bytes(), [1 + 3 + 5 + 7, 2 + 4 + 6 + 8])

    def test_multinode_single_p_assigns_only_ready_requested_remote_host(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router.prefill_urls = ["http://p0"]
        router._multinode_static_recovery = True
        router._multinode_recovery_parents = {"ready": 1, "writing": 1, "evicted": 1, "assigned": 1}
        entries = {
            key: {"source_host_node": "node-d", "state": state.value}
            for key, state in (("ready", HostStageState.HOST_READY),
                               ("tool_wait", HostStageState.HOST_READY),
                               ("writing", HostStageState.HOST_WRITING),
                               ("evicted", HostStageState.RECOMPUTE_REQUIRED),
                               ("assigned", HostStageState.HOST_READY))
        }
        entries["assigned"]["recovery_domain"] = 0
        assignments = []
        router._d2p_host_ledger = types.SimpleNamespace(
            snapshot_entries=lambda: entries,
            assign_d2p_recovery_domain=lambda sid, domain: assignments.append((sid, domain)),
        )
        router._prefill_arena_bytes()
        self.assertEqual(assignments, [("ready", 0)])
        assignments.clear()
        router._multinode_static_recovery = False
        router._prefill_arena_bytes()
        self.assertEqual(assignments, [])

    def test_multinode_source_host_router_assignment_allows_foreign_tp_pin(self):
        for tp in (1, 8):
            with self.subTest(tp=tp), tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
                router = self.make_router(Path(directory) / "ready")
                router.prefill_urls = ["http://p0"]
                router._multinode_static_recovery = True
                router._multinode_recovery_parents = {"source:g0": 1}
                ledger = SharedHostStagingLedger(str(Path(directory) / "host.json"))
                router._d2p_host_ledger = ledger

                def seed(entries):
                    entries["source:g0"] = {
                        "snapshot_id": "source:g0", "state": HostStageState.HOST_READY.value,
                        "p_owner": "d-host:decode-0", "tp_size": tp,
                        "source_host_node": "node-d", "arena_domain": 0,
                    }
                    return True, True

                ledger._mutate(seed, event_snapshot_id="source:g0")
                kwargs = dict(tp_rank=0, tp_size=tp, claim_id="restore0", recovery_domain=0)
                self.assertFalse(ledger.claim_d2p_recovery_rank("source:g0", "prefill-0", **kwargs))
                router._prefill_arena_bytes()
                self.assertEqual(ledger.get("source:g0")["recovery_domain"], 0)
                for rank in range(tp):
                    kwargs["tp_rank"] = rank
                    self.assertTrue(ledger.claim_d2p_recovery_rank("source:g0", "prefill-0", **kwargs))
                entry = ledger.get("source:g0")
                self.assertEqual(entry["p_owner"], "d-host:decode-0")
                self.assertEqual(entry["recovery_owner"], "prefill-0")
                self.assertEqual(entry["recovery_claim_id"], "restore0")
                self.assertEqual(entry["state"], HostStageState.H2D_LOADING.value)
                for rank in range(tp):
                    claim = entry["recovery_claims"][str(rank)]
                    self.assertEqual(claim["claim_id"], "restore0")
                    self.assertEqual(claim["phase"], "pinned")
                self.assertEqual(len(entry["recovery_claims"]), tp)

    async def test_p2d_pressure_counts_offered_and_durable_owner_once(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        room = 73
        snapshot_id = p2d_snapshot_id(room)
        admission = _PReadyAdmission(
            domain=0,
            sequence=1,
            submitted_key="p2d-pressure",
            enqueued_at=time.monotonic(),
            dispatch=AsyncMock(),
            future=asyncio.get_running_loop().create_future(),
            finished=asyncio.Event(),
            rooms=(room,),
            prompt_tokens=12_000,
        )
        router._p_ready_fifo_active = {0: {1: admission}}
        entry = {
            "state": HostStageState.OFFERED.value,
            "prefill_domain": 0,
            "token_count": 12_000,
            "byte_size": 100,
        }
        router.p2d_host_ledger = types.SimpleNamespace(
            snapshot_entries=lambda: {snapshot_id: dict(entry)}
        )

        offered = router._p2d_pressure_by_domain()[0]
        self.assertEqual(offered["p2d_inflight_tokens"], 12_000)
        self.assertEqual(offered["p2d_host_tokens"], 0)

        entry["state"] = HostStageState.HOST_RESERVED.value
        durable = router._p2d_pressure_by_domain()[0]
        self.assertEqual(durable["p2d_inflight_tokens"], 0)
        self.assertEqual(durable["p2d_host_tokens"], 12_000)

        entry["state"] = HostStageState.REJECTED.value
        rejected = router._p2d_pressure_by_domain()[0]
        self.assertEqual(rejected["p2d_inflight_tokens"], 12_000)
        self.assertEqual(rejected["p2d_host_tokens"], 0)

        router.p2d_host_ledger = types.SimpleNamespace(
            snapshot_entries=lambda: self.fail(
                "pre-fetched ledger rows must stay off the event loop"
            )
        )
        prefetched = router._p2d_pressure_by_domain(
            {
                snapshot_id: {
                    **entry,
                    "state": HostStageState.HOST_READY.value,
                }
            }
        )[0]
        self.assertEqual(prefetched["p2d_host_tokens"], 12_000)

    def test_cross_process_host_reservation_prevents_stale_herd(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            ledger = SharedPrefillPressureReservations(
                str(Path(directory) / "pressure-reservations.json"),
                ttl_seconds=10.0,
            )
            domains = [
                {
                    "domain": 0,
                    "hbm_capacity_tokens": 100,
                    "hbm_used_tokens": 0,
                    "arena_capacity_bytes": 100,
                    "arena_used_bytes": 0,
                },
                {
                    "domain": 1,
                    "hbm_capacity_tokens": 100,
                    "hbm_used_tokens": 0,
                    "arena_capacity_bytes": 100,
                    "arena_used_bytes": 0,
                },
            ]
            first = ledger.select_and_reserve("snapshot-a", 90, domains)
            second = ledger.select_and_reserve("snapshot-b", 90, domains)
            self.assertEqual((first, second), (0, 1))
            self.assertEqual(ledger.select_and_reserve("snapshot-a", 90, domains), 0)

    def test_host_reservation_v1_token_schema_is_cleared(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            path = Path(directory) / "pressure-reservations.json"
            path.write_bytes(
                orjson.dumps(
                    {
                        "version": 1,
                        "reservations": {
                            "old": {
                                "domain": 1,
                                "token_count": 9000,
                                "expires_at": time.time() + 60,
                            }
                        },
                    }
                )
            )
            ledger = SharedPrefillPressureReservations(str(path))

            self.assertEqual(ledger.totals(), {})
            self.assertEqual(orjson.loads(path.read_bytes())["version"], 2)

    async def test_backend_http_session_is_one_shared_pool(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        first = router._backend_http_session()
        second = router._backend_http_session()
        control = router._load_http_session()

        self.assertIs(first, second)
        self.assertIsNot(first, control)
        self.assertIs(control, router._load_http_session())
        self.assertTrue(control.connector.force_close)
        self.assertFalse(first.connector.force_close)
        self.assertEqual(first.connector._keepalive_timeout, 30.0)
        self.assertFalse(first.closed)
        await router.close()
        self.assertTrue(first.closed)
        self.assertTrue(control.closed)

    async def test_dynamic_prefill_shared_watcher_wakes_ready_batch(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
            router = self.make_router(Path(tmp))
            router.dynamic_prefill_domains = True
            unfinished = asyncio.get_running_loop().create_future()
            waits = [
                asyncio.create_task(
                    router._wait_until_prefill_ready_shared((room,), unfinished)
                )
                for room in (101, 102, 103)
            ]
            await asyncio.sleep(0)
            for sequence, room in enumerate((101, 102, 103)):
                (Path(tmp) / f"{room}.ready").write_bytes(
                    orjson.dumps(
                        {
                            "rid": f"r-{room}",
                            "num_kv_tokens": room,
                            "ready_sequence": sequence,
                            "prefill_domain": sequence % 2,
                        }
                    )
                )
            self.assertEqual(await asyncio.gather(*waits), [101, 102, 103])
            self.assertEqual(len(router._p_ready_snapshot), 3)
            router._p_ready_monitor_task.cancel()
            await asyncio.gather(
                router._p_ready_monitor_task, return_exceptions=True
            )

    def test_numa_domain_is_stable_and_decode_workers_are_partitioned(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router.numa_domains = True
        router.prefill_urls = ["http://p0:1", "http://p1:2"]
        router.prefill_bootstrap_ports = [100, 200]
        router.decode_urls = [f"http://d{index}" for index in range(6)]
        metadata = types.SimpleNamespace(request_id="stable-agent")

        first = router._request_domain(metadata, (1,))
        second = router._request_domain(metadata, (999,))

        self.assertEqual(first, second)
        self.assertEqual(
            router._domain_decode_urls(0),
            {"http://d0", "http://d1", "http://d2"},
        )
        self.assertEqual(
            router._domain_decode_urls(1),
            {"http://d3", "http://d4", "http://d5"},
        )

    def test_bind_prefill_domain_rewrites_bootstrap_destination(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router.prefill_urls = ["http://127.0.0.1:10", "http://127.0.0.2:20"]
        router.prefill_bootstrap_ports = [101, 202]
        request = {"bootstrap_room": 7}

        selected = router._bind_prefill_domain(request, 1)

        self.assertEqual(selected, "http://127.0.0.2:20")
        self.assertEqual(request["bootstrap_host"], "127.0.0.2")
        self.assertEqual(request["bootstrap_port"], 202)

    def test_global_decode_mode_keeps_slow_path_local_but_opens_p_to_all_d(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router.numa_domains = True
        router.global_decode = True
        router.prefill_urls = ["http://p0", "http://p1"]
        router.decode_urls = [f"http://d{index}" for index in range(6)]

        self.assertEqual(
            router._domain_decode_urls(0), set(router.decode_urls)
        )
        self.assertEqual(
            router._domain_decode_urls(1), set(router.decode_urls)
        )

    async def test_parent_direct_chooses_largest_remaining_kv(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router.numa_domains = True
            router.prefill_urls = ["http://p0", "http://p1"]
            router._prefill_pending_tokens = [9000, 1000]
            router._prefill_pressure_domains = [
                {
                    "domain": 0,
                    "hbm_used_tokens": 1000,
                    "hbm_capacity_tokens": 10000,
                },
                {
                    "domain": 1,
                    "hbm_used_tokens": 4000,
                    "hbm_capacity_tokens": 10000,
                },
            ]
            router._prefill_pressure_at = time.monotonic()
            router.early_claim_store = AgenticEarlyClaimStore(
                str(Path(directory) / "claims")
            )
            metadata = AgenticRequestMetadata(
                request_id="fixed-numa",
                generation=2,
                parent_generation=1,
            )
            request = {
                "bootstrap_room": 7,
                "sampling_params": {
                    "custom_params": {
                        "agentic_request_id": "fixed-numa",
                        "agentic_generation": 2,
                        "agentic_parent_generation": 1,
                    }
                },
            }
            original = time.time() - 0.1
            router.early_claim_store.publish_route(
                RequestGeneration("fixed-numa", 1),
                route="direct_ready",
                prefill_domain=0,
                snapshot_tokens=7000,
            )

            request["input_ids"] = list(range(7600))
            selected = await router._resolve_dynamic_prefill_work(
                request, metadata, original
            )
            self.assertTrue(selected.route_pending)
            # P selection only creates Router shadow accounting.  The
            # physical Direct grant is published by _late_dispatch as soon as
            # the target is known, independently of HTTP admission.
            self.assertIsNone(
                router.early_claim_store.read_arrival(
                    RequestGeneration("fixed-numa", 1),
                    not_before=0.0,
                    max_age_seconds=5.0,
                )
            )
            router._publish_parent_arrival(
                request,
                target_prefill_domain=selected.domain,
                arrived_at=original,
            )
            router.early_claim_store.publish_route(
                RequestGeneration("fixed-numa", 1),
                route="direct_complete",
                prefill_domain=1,
                snapshot_tokens=7000,
            )

            self.assertEqual(selected.domain, 0)
            self.assertEqual(selected.tokens, 600)
            self.assertEqual(router._prefill_pending_tokens, [9600, 1000])
            arrival = router.early_claim_store.read_arrival(
                RequestGeneration("fixed-numa", 1),
                not_before=0.0,
                max_age_seconds=5.0,
            )
            self.assertEqual(arrival["target_prefill_domain"], 0)
            self.assertEqual(arrival["arrived_at"], original)
            await router._release_prefill_work(selected)
            self.assertEqual(router._prefill_pending_tokens, [9000, 1000])

    async def test_parent_direct_excludes_p_without_complete_workset_capacity(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router._prefill_pending_tokens = [0, 5000]
            router._prefill_pressure_domains = [
                {
                    "domain": 0,
                    "hbm_used_tokens": 4000,
                    "hbm_capacity_tokens": 10000,
                },
                {
                    "domain": 1,
                    "hbm_used_tokens": 0,
                    "hbm_capacity_tokens": 10000,
                },
            ]
            router._prefill_pressure_at = time.monotonic()

            selected = await router._reserve_prefill_work(
                1000, direct_workset_tokens=8000
            )

            self.assertEqual(selected.domain, 1)
            self.assertEqual(router._prefill_direct_pending_tokens, [0, 8000])
            await router._release_prefill_work(selected)
            self.assertEqual(router._prefill_direct_pending_tokens, [0, 0])

    async def test_parent_direct_shadow_credit_covers_pressure_snapshot_lag(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router._prefill_pending_tokens = [0, 10000]
            router._prefill_pressure_domains = [
                {
                    "domain": domain,
                    "hbm_used_tokens": 0,
                    "hbm_capacity_tokens": 10000,
                }
                for domain in range(2)
            ]
            router._prefill_pressure_at = time.monotonic()

            first = await router._reserve_prefill_work(
                1000, direct_workset_tokens=8000
            )
            second = await router._reserve_prefill_work(
                1000, direct_workset_tokens=8000
            )

            self.assertEqual(first.domain, 0)
            self.assertEqual(second.domain, 1)
            self.assertEqual(router._prefill_direct_pending_tokens, [8000, 8000])
            await router._settle_direct_workset(first)
            self.assertEqual(router._prefill_direct_pending_tokens, [0, 8000])
            await router._release_prefill_work(first)
            await router._release_prefill_work(second)
            self.assertEqual(router._prefill_direct_pending_tokens, [0, 0])

    async def test_paired_prefill_credit_waits_for_real_capacity(self):
        router = self.make_router(Path("/dev/shm/test-paired-credit"))
        router._prefill_pressure_domains = [
            {"domain": 0, "hbm_used_tokens": 9000, "hbm_capacity_tokens": 10000},
            {"domain": 1, "hbm_used_tokens": 9000, "hbm_capacity_tokens": 10000},
        ]
        router._prefill_pressure_at = time.monotonic()

        pending = asyncio.create_task(
            router._reserve_paired_prefill_credit(
                {"input_ids": list(range(2000))}
            )
        )
        await asyncio.sleep(0.01)
        self.assertFalse(pending.done())
        self.assertEqual(router._prefill_direct_pending_tokens, [0, 0])

        router._prefill_pressure_domains[1]["hbm_used_tokens"] = 7000
        router._prefill_pressure_at = time.monotonic()
        reservation = await asyncio.wait_for(pending, timeout=1)
        self.assertEqual(reservation.domain, 1)
        self.assertEqual(router._prefill_direct_pending_tokens, [0, 2000])
        self.assertTrue(reservation.route_pending)
        await router._release_prefill_work(reservation)

    async def test_paired_prefill_credit_is_atomic_across_waiters(self):
        router = self.make_router(Path("/dev/shm/test-paired-credit-atomic"))
        router.prefill_urls = ["http://p0"]
        router._prefill_pending_tokens = [0]
        router._prefill_pending_requests = [0]
        router._prefill_direct_pending_tokens = [0]
        router._prefill_pressure_domains = [
            {"domain": 0, "hbm_used_tokens": 0, "hbm_capacity_tokens": 3000}
        ]
        router._prefill_pressure_at = time.monotonic()

        first_task = asyncio.create_task(
            router._reserve_paired_prefill_credit(
                {"input_ids": list(range(2000))}
            )
        )
        second_task = asyncio.create_task(
            router._reserve_paired_prefill_credit(
                {"input_ids": list(range(2000))}
            )
        )
        first = await asyncio.wait_for(first_task, timeout=1)
        await asyncio.sleep(0.01)
        self.assertFalse(second_task.done())
        await router._release_prefill_work(first)
        second = await asyncio.wait_for(second_task, timeout=1)
        self.assertEqual(second.domain, 0)
        await router._release_prefill_work(second)

    async def test_direct_shadow_hands_off_only_to_causally_new_pressure(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router._prefill_pending_tokens = [0, 10000]
            router._prefill_pressure_domains = [
                {
                    "domain": domain,
                    "hbm_used_tokens": 0,
                    "hbm_capacity_tokens": 10000,
                }
                for domain in range(2)
            ]
            router._prefill_pressure_at = time.monotonic()
            router._prefill_pressure_sample_started_at = time.monotonic()

            first = await router._reserve_prefill_work(
                1000, direct_workset_tokens=8000
            )
            terminal_at = time.monotonic()
            handoff = asyncio.create_task(
                router._settle_direct_workset_after_pressure(
                    first, direct_terminal_at=terminal_at
                )
            )
            await asyncio.sleep(0)

            # A stale physical sample must not combine with cleared shadow
            # credit and admit another workset to P0.
            third = await router._reserve_prefill_work(
                1000, direct_workset_tokens=8000
            )
            self.assertEqual(third.domain, 1)
            self.assertEqual(router._prefill_direct_pending_tokens, [8000, 8000])
            self.assertFalse(handoff.done())

            # Publish a sample fetched after Direct completion.  Its physical
            # used count now replaces, rather than overlaps or misses, shadow.
            router._prefill_pressure_domains = [
                {
                    "domain": 0,
                    "hbm_used_tokens": 8000,
                    "hbm_capacity_tokens": 10000,
                },
                {
                    "domain": 1,
                    "hbm_used_tokens": 0,
                    "hbm_capacity_tokens": 20000,
                },
            ]
            router._prefill_pressure_sample_started_at = terminal_at + 0.001
            router._prefill_pressure_at = time.monotonic()
            await asyncio.wait_for(handoff, timeout=1)
            self.assertEqual(router._prefill_direct_pending_tokens, [0, 8000])

            fourth = await router._reserve_prefill_work(
                1000, direct_workset_tokens=8000
            )
            self.assertEqual(fourth.domain, 1)
            await router._release_prefill_work(first)
            await router._release_prefill_work(third)
            await router._release_prefill_work(fourth)

    async def test_slow_shadow_hands_off_only_after_scheduled_pressure(self):
        router = self.make_router(Path("/dev/shm/test-slow-shadow"))
        router._prefill_pressure_domains = [
            {
                "domain": domain,
                "hbm_used_tokens": 0,
                "hbm_capacity_tokens": 20_000,
            }
            for domain in range(2)
        ]
        router._prefill_pressure_at = time.monotonic()
        router._prefill_pressure_sample_started_at = time.monotonic()
        slow = await router._reserve_prefill_work(
            1000, domain=0, direct_workset_tokens=8000
        )
        scheduled_at = time.monotonic()
        handoff = asyncio.create_task(
            router._settle_direct_workset_after_pressure(
                slow, direct_terminal_at=scheduled_at
            )
        )
        await asyncio.sleep(0)

        self.assertFalse(handoff.done())
        self.assertEqual(router._prefill_direct_pending_tokens, [8000, 0])

        router._prefill_pressure_domains[0]["hbm_used_tokens"] = 8000
        router._prefill_pressure_sample_started_at = scheduled_at + 0.001
        router._prefill_pressure_at = time.monotonic()
        await asyncio.wait_for(handoff, timeout=1)

        self.assertEqual(router._prefill_direct_pending_tokens, [0, 0])
        await router._release_prefill_work(slow)

    async def test_direct_fallback_retargets_after_host_is_durable(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router.numa_domains = True
            router.early_claim_store = AgenticEarlyClaimStore(
                str(Path(directory) / "claims")
            )
            router._prefill_pressure_at = time.monotonic()
            router._prefill_pressure_domains = [
                {"domain": 0, "hbm_capacity_tokens": 20_000, "hbm_used_tokens": 0},
                {"domain": 1, "hbm_capacity_tokens": 20_000, "hbm_used_tokens": 15_000},
            ]
            router._prefill_pending_tokens = [8000, 0]
            host_entry = {
                "state": HostStageState.HOST_READY.value,
                "arena_domain": 0,
            }

            def assign_recovery(_snapshot_id, domain):
                host_entry["recovery_domain"] = int(domain)
                return True

            router._d2p_host_ledger = types.SimpleNamespace(
                get=lambda _snapshot_id: dict(host_entry),
                assign_d2p_recovery_domain=assign_recovery,
                snapshot_entries=lambda: {"host": dict(host_entry)},
            )
            metadata = AgenticRequestMetadata(
                request_id="fallback", generation=1, parent_generation=0
            )
            request = {
                "bootstrap_room": 9,
                "input_ids": list(range(9000)),
                "sampling_params": {
                    "custom_params": {
                        "agentic_request_id": "fallback",
                        "agentic_generation": 1,
                        "agentic_parent_generation": 0,
                    }
                },
            }
            parent = RequestGeneration("fallback", 0)
            router.early_claim_store.publish_route(
                parent,
                route="direct_ready",
                prefill_domain=0,
                snapshot_tokens=8000,
            )

            selected = await router._resolve_dynamic_prefill_work(
                request, metadata, time.time()
            )
            self.assertIsNone(
                router.early_claim_store.read_arrival(
                    parent, not_before=0.0, max_age_seconds=5.0
                )
            )
            router._publish_parent_arrival(
                request,
                target_prefill_domain=selected.domain,
                arrived_at=time.time(),
            )
            watcher = asyncio.create_task(
                router._watch_dynamic_prefill_route(
                    request, metadata, selected
                )
            )
            router._prefill_pressure_at = time.monotonic()
            router._prefill_pressure_domains = [
                {"domain": 0, "hbm_capacity_tokens": 20_000, "hbm_used_tokens": 15_000},
                {"domain": 1, "hbm_capacity_tokens": 20_000, "hbm_used_tokens": 0},
            ]
            router.early_claim_store.publish_route(
                parent,
                route="host_ready",
                prefill_domain=-1,
                arena_domain=0,
                arena_numa_node=0,
                snapshot_tokens=8000,
            )
            outcome = await asyncio.wait_for(watcher, timeout=1)

            self.assertEqual(selected.domain, 1)
            self.assertEqual(outcome["action"], "redirect")
            self.assertEqual(outcome["route"], "host_ready")
            self.assertEqual(selected.tokens, 1000)
            self.assertEqual(router._prefill_pending_tokens, [8000, 1000])
            await router._move_prefill_work(selected, 0)
            router._publish_parent_arrival(
                request, target_prefill_domain=0, arrived_at=time.time()
            )
            self.assertEqual(router._prefill_pending_tokens, [9000, 0])
            arrival = router.early_claim_store.read_arrival(
                parent, not_before=0.0, max_age_seconds=5.0
            )
            self.assertEqual(arrival["target_prefill_domain"], 0)
            await router._release_prefill_work(selected)

    async def test_recompute_route_charges_full_prompt_tokens(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router.numa_domains = True
            router.early_claim_store = AgenticEarlyClaimStore(
                str(Path(directory) / "claims")
            )
            metadata = AgenticRequestMetadata(
                request_id="recompute", generation=1, parent_generation=0
            )
            request = {
                "bootstrap_room": 10,
                "input_ids": list(range(9000)),
                "sampling_params": {
                    "custom_params": {
                        "agentic_request_id": "recompute",
                        "agentic_generation": 1,
                        "agentic_parent_generation": 0,
                    }
                },
            }
            parent = RequestGeneration("recompute", 0)
            router.early_claim_store.publish_route(
                parent,
                route="direct_ready",
                prefill_domain=0,
                snapshot_tokens=8000,
            )
            selected = await router._resolve_dynamic_prefill_work(
                request, metadata, time.time()
            )
            self.assertEqual(selected.tokens, 1000)

            watcher = asyncio.create_task(
                router._watch_dynamic_prefill_route(
                    request, metadata, selected
                )
            )
            router.early_claim_store.publish_route(
                parent,
                route="recompute",
                prefill_domain=selected.domain,
                snapshot_tokens=8000,
            )
            outcome = await asyncio.wait_for(watcher, timeout=1)

            self.assertEqual(outcome["action"], "recompute")
            self.assertEqual(selected.tokens, 9000)
            self.assertEqual(
                router._prefill_pending_tokens[selected.domain], 9000
            )
            await router._release_prefill_work(selected)

    async def test_evicted_host_route_charges_full_prompt_without_timeout(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router.numa_domains = True
            router.early_claim_store = AgenticEarlyClaimStore(
                str(Path(directory) / "claims")
            )
            ledger_path = str(Path(directory) / "host-ledger.json")
            router._d2p_host_ledger = SharedHostStagingLedger(ledger_path)
            metadata = AgenticRequestMetadata(
                request_id="host-evicted", generation=1, parent_generation=0
            )
            request = {
                "bootstrap_room": 11,
                "input_ids": list(range(9000)),
                "sampling_params": {
                    "custom_params": {
                        "agentic_request_id": "host-evicted",
                        "agentic_generation": 1,
                        "agentic_parent_generation": 0,
                    }
                },
            }
            parent = RequestGeneration("host-evicted", 0)

            def publish_evicted(entries):
                entries[parent.snapshot_id] = {
                    "snapshot_id": parent.snapshot_id,
                    "state": HostStageState.RECOMPUTE_REQUIRED.value,
                    "p_owner": "p0",
                    "tp_size": 1,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }
                return True, True

            router._d2p_host_ledger._mutate(
                publish_evicted, event_snapshot_id=parent.snapshot_id
            )
            router.early_claim_store.publish_route(
                parent,
                route="host_ready",
                prefill_domain=0,
                arena_numa_node=0,
                snapshot_tokens=8000,
            )

            selected = await router._resolve_dynamic_prefill_work(
                request, metadata, time.time()
            )

            self.assertEqual(selected.tokens, 9000)
            self.assertEqual(
                router.early_claim_store.read_route(parent)["route"], "recompute"
            )
            await router._release_prefill_work(selected)

    async def test_dynamic_prefill_fifo_is_independent_per_p(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.dynamic_prefill_domains = True
            router._ready_path(1).write_bytes(
                orjson.dumps(
                    {
                        "num_kv_tokens": 10,
                        "ready_sequence": 0,
                        "prefill_domain": 0,
                    }
                )
            )
            router._ready_path(2).write_bytes(
                orjson.dumps(
                    {
                        "num_kv_tokens": 10,
                        "ready_sequence": 0,
                        "prefill_domain": 1,
                    }
                )
            )

            entered = [asyncio.Event(), asyncio.Event()]
            release = asyncio.Event()

            async def dispatch(domain):
                entered[domain].set()
                await release.wait()
                return domain

            task0 = asyncio.create_task(
                router._dispatch_p_ready_in_order(0, 0, lambda: dispatch(0))
            )
            task1 = asyncio.create_task(
                router._dispatch_p_ready_in_order(0, 1, lambda: dispatch(1))
            )
            await asyncio.gather(*(event.wait() for event in entered))
            self.assertIn((0, 0), router._p_ready_submitted_sequences)
            self.assertIn((1, 0), router._p_ready_submitted_sequences)
            release.set()
            self.assertEqual(await asyncio.gather(task0, task1), [0, 1])
            await router.close()

    async def test_ready_marker_reports_exact_prompt_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            room = 17
            router._ready_path(room).write_bytes(
                orjson.dumps({"rid": "request-a", "num_kv_tokens": 7317})
            )
            unfinished = asyncio.create_task(asyncio.sleep(10))
            try:
                tokens = await router._wait_until_prefill_ready((room,), unfinished)
            finally:
                unfinished.cancel()
            self.assertEqual(tokens, 7317)

    async def test_p_ready_fifo_uses_p_completion_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router._ready_path(1).write_bytes(
                orjson.dumps({"num_kv_tokens": 10, "ready_sequence": 4})
            )
            router._ready_path(2).write_bytes(
                orjson.dumps({"num_kv_tokens": 10, "ready_sequence": 5})
            )

            order = []

            async def dispatch(sequence):
                order.append(sequence)
                return sequence

            later = asyncio.create_task(
                router._dispatch_p_ready_in_order(5, 0, lambda: dispatch(5))
            )
            await asyncio.sleep(0.01)
            self.assertFalse(later.done())

            first = asyncio.create_task(
                router._dispatch_p_ready_in_order(4, 0, lambda: dispatch(4))
            )
            self.assertEqual(await asyncio.wait_for(first, timeout=1), 4)
            self.assertEqual(await asyncio.wait_for(later, timeout=1), 5)
            self.assertEqual(order, [4, 5])
            await router.close()

    async def test_p_ready_fifo_ignores_health_probe_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router._ready_path(0).write_bytes(
                orjson.dumps(
                    {
                        "rid": "HEALTH_CHECK_123",
                        "num_kv_tokens": 1,
                        "ready_sequence": 2,
                    }
                )
            )
            router._ready_path(1).write_bytes(
                orjson.dumps(
                    {
                        "rid": "workload-request",
                        "num_kv_tokens": 10,
                        "ready_sequence": 3,
                    }
                )
            )

            async def dispatch():
                return 3

            self.assertEqual(
                await asyncio.wait_for(
                    router._dispatch_p_ready_in_order(3, 0, dispatch), timeout=1
                ),
                3,
            )
            await router.close()

    async def test_p_ready_fifo_ignores_room_zero_with_random_probe_rid(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router._ready_path(0).write_bytes(
                orjson.dumps(
                    {
                        "rid": "probe-with-random-id",
                        "num_kv_tokens": 1,
                        "ready_sequence": 0,
                    }
                )
            )
            router._ready_path(1).write_bytes(
                orjson.dumps(
                    {
                        "rid": "workload-request",
                        "num_kv_tokens": 10,
                        "ready_sequence": 1,
                    }
                )
            )

            async def dispatch():
                return 1

            self.assertEqual(
                await asyncio.wait_for(
                    router._dispatch_p_ready_in_order(1, 0, dispatch), timeout=1
                ),
                1,
            )
            await router.close()

    async def test_p_ready_fifo_ignores_late_marker_from_redirected_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.dynamic_prefill_domains = True
            router._active_prefill_attempts = {22: "current-rid"}
            router._p_ready_snapshot = {
                11: {
                    "_path": router._ready_path(11),
                    "rid": "aborted-rid",
                    "prefill_domain": 0,
                    "ready_sequence": 4,
                },
                22: {
                    "_path": router._ready_path(22),
                    "rid": "current-rid",
                    "prefill_domain": 0,
                    "ready_sequence": 5,
                },
            }
            router._ready_path(11).write_bytes(
                orjson.dumps(router._p_ready_snapshot[11] | {"_path": None})
            )
            router._ready_path(22).write_bytes(
                orjson.dumps(router._p_ready_snapshot[22] | {"_path": None})
            )

            async def dispatch():
                return 5

            self.assertEqual(
                await asyncio.wait_for(
                    router._dispatch_p_ready_in_order(5, 0, dispatch), timeout=1
                ),
                5,
            )
            self.assertFalse(router._ready_path(11).exists())
            self.assertTrue(router._ready_path(22).exists())
            await router.close()

    async def test_p_ready_fifo_still_orders_two_owned_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.dynamic_prefill_domains = True
            router._active_prefill_attempts = {31: "first", 32: "second"}
            for room, rid, sequence in ((31, "first", 4), (32, "second", 5)):
                payload = {
                    "rid": rid,
                    "prefill_domain": 0,
                    "ready_sequence": sequence,
                }
                path = router._ready_path(room)
                path.write_bytes(orjson.dumps(payload))
                router._p_ready_snapshot[room] = payload | {"_path": path}

            order = []

            async def dispatch(sequence):
                order.append(sequence)
                return sequence

            later = asyncio.create_task(
                router._dispatch_p_ready_in_order(5, 0, lambda: dispatch(5))
            )
            await asyncio.sleep(0.01)
            self.assertFalse(later.done())
            first = asyncio.create_task(
                router._dispatch_p_ready_in_order(4, 0, lambda: dispatch(4))
            )
            self.assertEqual(await asyncio.wait_for(first, timeout=1), 4)
            self.assertEqual(await asyncio.wait_for(later, timeout=1), 5)
            self.assertEqual(order, [4, 5])
            await router.close()

    async def test_p_ready_fifo_has_one_order_scan_per_submission_not_per_waiter(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.dynamic_prefill_domains = True
            router._active_prefill_attempts = {}
            for sequence in range(64):
                room = 10_000 + sequence
                rid = f"rid-{sequence}"
                payload = {
                    "rid": rid,
                    "prefill_domain": 0,
                    "ready_sequence": sequence,
                }
                path = router._ready_path(room)
                path.write_bytes(orjson.dumps(payload))
                router._p_ready_snapshot[room] = payload | {"_path": path}
                router._active_prefill_attempts[room] = rid

            original_oldest = router._oldest_p_ready_sequence
            scans = 0

            def counted_oldest(domain=0):
                nonlocal scans
                scans += 1
                return original_oldest(domain)

            router._oldest_p_ready_sequence = counted_oldest
            tasks = {
                sequence: asyncio.create_task(
                    router._dispatch_p_ready_in_order(
                        sequence,
                        0,
                        lambda value=sequence: asyncio.sleep(0, result=value),
                    )
                )
                for sequence in reversed(range(64))
            }
            for sequence in range(64):
                self.assertEqual(
                    await asyncio.wait_for(tasks[sequence], timeout=1), sequence
                )

            # A per-waiter polling design performs thousands of full scans for
            # this burst. The dispatcher performs approximately one per item.
            self.assertLessEqual(scans, 128)
            await router.close()

    async def test_p_ready_broker_pipelines_fifo_heads_without_overcommit(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.dynamic_prefill_domains = True
            router.global_decode = True
            router.numa_domains = True
            router._active_prefill_attempts = {}
            router._p2d_host_offered_snapshots = set()
            router._p_ready_admission_window_per_p = 4
            loads = [
                DecodeLoad(
                    url=url,
                    used_tokens=0,
                    capacity_tokens=4096,
                    running=0,
                    waiting=0,
                    prealloc=0,
                    transfer=0,
                    max_running=128,
                )
                for url in router.decode_urls
            ]
            router._all_decode_loads = AsyncMock(return_value=loads)
            router._load_http_session = lambda: object()

            release_first = asyncio.Event()
            commit_started = asyncio.Event()
            committed = []

            async def submit(domain, sequence, room):
                rid = f"p{domain}-s{sequence}"
                payload = {
                    "rid": rid,
                    "prefill_domain": domain,
                    "ready_sequence": sequence,
                }
                router._ready_path(room).write_bytes(orjson.dumps(payload))
                router._p_ready_snapshot[room] = payload | {
                    "_path": router._ready_path(room)
                }
                router._active_prefill_attempts[room] = rid

                async def fallback():
                    self.fail("ordinary Direct admission used fallback selector")

                async def commit(reservation):
                    committed.append((domain, sequence, reservation.url))
                    if sequence == 1:
                        commit_started.set()
                        await release_first.wait()
                    return reservation

                return await router._dispatch_p_ready_in_order(
                    sequence,
                    domain,
                    fallback,
                    request={"sampling_params": {"max_new_tokens": 512}},
                    rooms=(room,),
                    prompt_tokens=512,
                    commit=commit,
                )

            tasks = [
                asyncio.create_task(submit(domain, sequence, 100 + domain * 10 + sequence))
                for domain in (0, 1)
                for sequence in (1, 2)
            ]
            await asyncio.wait_for(commit_started.wait(), timeout=1)
            deadline = time.monotonic() + 1
            while len(router._reservations) < 4 and time.monotonic() < deadline:
                await asyncio.sleep(0.001)
            self.assertEqual(len(router._reservations), 4)
            for url in router.decode_urls:
                reserved = sum(
                    item.admission_tokens
                    for item in router._reservations.values()
                    if item.url == url
                )
                self.assertLessEqual(reserved, 4096)

            release_first.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
            for domain in (0, 1):
                self.assertEqual(
                    [sequence for p, sequence, _url in committed if p == domain],
                    [1, 2],
                )
            await router.close()

    async def test_p_ready_host_staging_releases_producer_before_d_capacity(self):
        """Blocked requests stage independently without pinning P HBM."""

        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.dynamic_prefill_domains = True
            router._active_prefill_attempts = {}
            router._p_ready_stage_lanes_per_p = 2

            async def no_direct(admissions):
                return {admission: None for admission in admissions}

            router._reserve_p_ready_direct_batch = no_direct
            staged = []
            committed = []
            first_commit_started = asyncio.Event()
            release_first_commit = asyncio.Event()

            async def submit(sequence, room):
                rid = f"staged-{sequence}"
                payload = {
                    "rid": rid,
                    "prefill_domain": 0,
                    "ready_sequence": sequence,
                }
                path = router._ready_path(room)
                path.write_bytes(orjson.dumps(payload))
                router._p_ready_snapshot[room] = payload | {"_path": path}
                router._active_prefill_attempts[room] = rid

                async def prepare():
                    staged.append(sequence)
                    await asyncio.sleep(0)
                    # None means a complete Host snapshot now owns the KV, so
                    # the P scheduler may release this generation immediately.
                    return None

                async def dispatch():
                    committed.append(sequence)
                    if sequence == 1:
                        first_commit_started.set()
                        await release_first_commit.wait()
                    return sequence

                return await router._dispatch_p_ready_in_order(
                    sequence,
                    0,
                    dispatch,
                    request={"max_tokens": 512},
                    rooms=(room,),
                    prompt_tokens=512,
                    commit=lambda reservation: asyncio.sleep(
                        0, result=reservation
                    ),
                    prepare=prepare,
                )

            first = asyncio.create_task(submit(1, 501))
            second = asyncio.create_task(submit(2, 502))
            await asyncio.wait_for(first_commit_started.wait(), timeout=1)
            deadline = time.monotonic() + 1
            while len(staged) < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.001)

            self.assertEqual(staged, [1, 2])
            self.assertEqual(committed, [1])
            release_first_commit.set()
            self.assertEqual(
                await asyncio.wait_for(asyncio.gather(first, second), timeout=1),
                [1, 2],
            )
            self.assertEqual(committed, [1, 2])
            await router.close()

    async def test_p_ready_grace_retries_direct_before_host_staging(self):
        """Capacity released during the grace interval must avoid Host."""

        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.dynamic_prefill_domains = True
            router._active_prefill_attempts = {}
            router.p2d_host_spill_delay = 0.001

            async def no_initial_direct(admissions):
                return {admission: None for admission in admissions}

            reservation = DecodeReservation(
                reservation_id="after-grace",
                url="http://d0",
                prompt_tokens=512,
                admission_tokens=1024,
                request_count=1,
                rooms=(505,),
                created_at=time.monotonic(),
                prefill_domain=0,
            )
            router._reserve_p_ready_direct_batch = no_initial_direct
            router._retry_p_ready_direct_after_grace = AsyncMock(
                return_value=reservation
            )
            prepare = AsyncMock(side_effect=AssertionError("must not stage"))
            dispatch = AsyncMock(side_effect=AssertionError("must not reselect"))

            rid = "grace-retry"
            payload = {"rid": rid, "prefill_domain": 0, "ready_sequence": 1}
            path = router._ready_path(505)
            path.write_bytes(orjson.dumps(payload))
            router._p_ready_snapshot[505] = payload | {"_path": path}
            router._active_prefill_attempts[505] = rid

            result = await router._dispatch_p_ready_in_order(
                1,
                0,
                dispatch,
                request={"max_tokens": 512},
                rooms=(505,),
                prompt_tokens=512,
                commit=lambda selected: asyncio.sleep(0, result=selected),
                prepare=prepare,
            )

            self.assertIs(result, reservation)
            router._retry_p_ready_direct_after_grace.assert_awaited_once()
            prepare.assert_not_awaited()
            dispatch.assert_not_awaited()
            await router.close()

    async def test_p_ready_grace_retry_allows_later_feasible_request(self):
        """An older blocked generation must not gate a later Direct retry."""

        router = self.make_router(Path("/dev/shm/test-ready-grace-fifo"))
        loop = asyncio.get_running_loop()

        def admission(sequence):
            return _PReadyAdmission(
                domain=0,
                sequence=sequence,
                submitted_key=(0, sequence),
                enqueued_at=time.monotonic(),
                dispatch=AsyncMock(),
                future=loop.create_future(),
                finished=asyncio.Event(),
                request={"max_tokens": 512},
                rooms=(500 + sequence,),
                prompt_tokens=512,
                commit=AsyncMock(),
                prepare=AsyncMock(),
            )

        first = admission(1)
        second = admission(2)
        router._p_ready_fifo_active = {0: {1: first, 2: second}}
        router._load_cache = [
            DecodeLoad("http://d0", 0, 4096, 0, 0, 0, 0, 0),
            DecodeLoad("http://d1", 0, 4096, 0, 0, 0, 0, 0),
        ]
        router._observe_decode_load_after = AsyncMock(return_value=True)

        selected = await router._retry_p_ready_direct_after_grace(
            second, not_before=time.monotonic()
        )
        self.assertIsNotNone(selected)
        self.assertEqual(len(router._reservations), 1)
        router._observe_decode_load_after.assert_awaited_once()

    async def test_stage_returns_at_host_durability_without_d_capacity(self):
        """Host durability, not D admission, is the P-HBM release boundary."""

        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.global_decode = True
            router.p2d_host_spill_delay = 0.0
            snapshot_id = p2d_snapshot_id(503)
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            router._p2d_host_offered_snapshots = set()

            staged = asyncio.create_task(
                router._stage_p2d_until_durable((503,), 5_000, 0)
            )
            deadline = time.monotonic() + 1
            while ledger.get(snapshot_id) is None and time.monotonic() < deadline:
                await asyncio.sleep(0.001)

            owner = "p2d-p:test"
            self.assertIsNotNone(ledger.claim(snapshot_id, owner))
            self.assertTrue(
                ledger.publish_grants(
                    snapshot_id, owner, [{"kind": "shared_host_extent"}]
                )
            )
            self.assertTrue(ledger.ack_chunk(snapshot_id, owner, 0))
            self.assertTrue(ledger.mark_host_ready(snapshot_id, owner, 1))

            self.assertTrue(await asyncio.wait_for(staged, timeout=1))
            self.assertEqual(router._reservations, {})

    async def test_stage_rejected_without_host_owner_retains_p_for_direct(self):
        """Arena-capacity rejection is RETAIN_P, not a stuck/error state."""

        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.global_decode = True
            router.p2d_host_spill_delay = 0.0
            snapshot_id = p2d_snapshot_id(504)
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            router._p2d_host_offered_snapshots = set()

            staged = asyncio.create_task(
                router._stage_p2d_until_durable((504,), 5_000, 0)
            )
            deadline = time.monotonic() + 1
            while ledger.get(snapshot_id) is None and time.monotonic() < deadline:
                await asyncio.sleep(0.001)
            self.assertTrue(
                ledger.reject_unclaimed_offer(
                    snapshot_id, reason="p2d_host_capacity"
                )
            )

            self.assertFalse(await asyncio.wait_for(staged, timeout=1))
            self.assertEqual(
                ledger.get(snapshot_id)["state"], HostStageState.REJECTED.value
            )

    async def test_p_ready_commit_does_not_wait_for_older_generation(self):
        """Cross-request FIFO metadata cannot block a feasible Direct send."""

        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router._ensure_p_ready_admission_state()
            loop = asyncio.get_running_loop()
            predecessor = loop.create_future()
            reservation = DecodeReservation(
                reservation_id="shutdown-reservation",
                url="http://d0",
                prompt_tokens=512,
                admission_tokens=1024,
                request_count=1,
                rooms=(505,),
                created_at=time.monotonic(),
            )
            router._reservations[reservation.reservation_id] = reservation
            admission = late_binding_router_module._PReadyAdmission(
                domain=0,
                sequence=1,
                submitted_key=1,
                enqueued_at=time.monotonic(),
                dispatch=lambda: asyncio.sleep(0),
                future=loop.create_future(),
                finished=asyncio.Event(),
                rooms=(505,),
                commit=lambda value: asyncio.sleep(0, result=value),
                commit_predecessor=predecessor,
                commit_done=loop.create_future(),
            )
            router._p_ready_fifo_active = {0: {1: admission}}
            task = asyncio.create_task(router._run_p_ready_admission(admission, reservation))
            admission.dispatch_task = task
            await asyncio.wait_for(task, timeout=1)

            self.assertTrue(admission.commit_started)
            self.assertTrue(admission.finished.is_set())
            self.assertFalse(predecessor.done())
            router._reservations.pop(reservation.reservation_id, None)
            await router.close()

    async def test_p_ready_broker_cancellation_during_load_does_not_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.global_decode = True
            router.numa_domains = False
            router._p2d_host_offered_snapshots = set()
            payload = {"rid": "cancel", "ready_sequence": 1}
            router._ready_path(17).write_bytes(orjson.dumps(payload))
            load_started = asyncio.Event()
            release_load = asyncio.Event()

            async def loads(_session):
                load_started.set()
                await release_load.wait()
                return [
                    DecodeLoad(
                        "http://d0", 0, 100_000, 0, 0, 0, 0, 128
                    )
                ]

            router._all_decode_loads = loads
            router._load_http_session = lambda: object()
            caller = asyncio.create_task(
                router._dispatch_p_ready_in_order(
                    1,
                    0,
                    lambda: asyncio.sleep(0),
                    request={"max_tokens": 512},
                    rooms=(17,),
                    prompt_tokens=512,
                    commit=lambda reservation: asyncio.sleep(
                        0, result=reservation
                    ),
                )
            )
            await asyncio.wait_for(load_started.wait(), timeout=1)
            caller.cancel()
            release_load.set()
            with self.assertRaises(asyncio.CancelledError):
                await caller
            await asyncio.sleep(0)
            self.assertFalse(router._reservations)
            await router.close()

    async def test_p_ready_fifo_cancel_before_dispatch_removes_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            payload = {"rid": "first", "num_kv_tokens": 10, "ready_sequence": 6}
            router._ready_path(6).write_bytes(orjson.dumps(payload))
            router._ready_path(7).write_bytes(
                orjson.dumps(payload | {"rid": "cancel-me", "ready_sequence": 7})
            )
            blocker = asyncio.Event()

            async def block_first():
                await blocker.wait()

            first = asyncio.create_task(
                router._dispatch_p_ready_in_order(6, 0, block_first)
            )
            await asyncio.sleep(0)
            cancelled = asyncio.create_task(
                router._dispatch_p_ready_in_order(
                    7, 0, lambda: asyncio.sleep(0, result=7)
                )
            )
            await asyncio.sleep(0)
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled
            self.assertNotIn(7, router._p_ready_fifo_waiters[0])
            blocker.set()
            await first
            await router.close()

    async def test_p_ready_fifo_cancel_active_dispatch_finishes_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            payload = {"rid": "active", "num_kv_tokens": 10, "ready_sequence": 8}
            router._ready_path(8).write_bytes(orjson.dumps(payload))
            entered = asyncio.Event()
            allow_commit = asyncio.Event()
            physically_quiesced = asyncio.Event()
            submitted = []

            async def active_dispatch():
                try:
                    entered.set()
                    await allow_commit.wait()
                    submitted.append(True)
                finally:
                    # Models the selector reaching its atomic D-submit
                    # boundary before caller cancellation becomes visible.
                    await asyncio.sleep(0)
                    physically_quiesced.set()

            caller = asyncio.create_task(
                router._dispatch_p_ready_in_order(8, 0, active_dispatch)
            )
            await asyncio.wait_for(entered.wait(), timeout=1)
            caller.cancel()
            await asyncio.sleep(0)
            self.assertFalse(caller.done())
            allow_commit.set()
            with self.assertRaises(asyncio.CancelledError):
                await caller

            self.assertTrue(physically_quiesced.is_set())
            self.assertEqual(submitted, [True])
            self.assertNotIn(0, router._p_ready_fifo_active)
            self.assertIn(8, router._p_ready_submitted_sequences)
            await router.close()

    async def test_p_ready_fifo_shutdown_joins_active_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            payload = {"rid": "shutdown", "num_kv_tokens": 10, "ready_sequence": 9}
            router._ready_path(9).write_bytes(orjson.dumps(payload))
            entered = asyncio.Event()
            physically_quiesced = asyncio.Event()

            async def active_dispatch():
                try:
                    entered.set()
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0)
                    physically_quiesced.set()

            caller = asyncio.create_task(
                router._dispatch_p_ready_in_order(9, 0, active_dispatch)
            )
            await asyncio.wait_for(entered.wait(), timeout=1)
            await router.close()

            self.assertTrue(physically_quiesced.is_set())
            self.assertNotIn(0, router._p_ready_fifo_active)
            self.assertNotIn(9, router._p_ready_submitted_sequences)
            with self.assertRaises(asyncio.CancelledError):
                await caller

    async def test_p_ready_fifo_scan_failure_fails_waiter(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router._oldest_p_ready_sequence = lambda _domain=0: (_ for _ in ()).throw(
                RuntimeError("scan failed")
            )

            with self.assertRaisesRegex(RuntimeError, "scan failed"):
                await router._dispatch_p_ready_in_order(
                    9, 0, lambda: asyncio.sleep(0)
                )
            await router.close()

    def test_prefill_attempt_ownership_tracks_all_tp_rooms_and_exact_rids(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router._active_prefill_attempts = {}
        request = {
            "bootstrap_room": [101, 102],
            "rid": ["rank-0", "rank-1"],
        }

        router._activate_prefill_attempt(request, (101, 102))
        self.assertEqual(
            router._active_prefill_attempts,
            {101: "rank-0", 102: "rank-1"},
        )
        self.assertTrue(
            router._p_ready_marker_is_owned(101, {"rid": "rank-0"})
        )
        self.assertFalse(
            router._p_ready_marker_is_owned(101, {"rid": "stale-rank-0"})
        )

        router._deactivate_prefill_attempt(request, (101, 102))
        self.assertEqual(router._active_prefill_attempts, {})

    def test_batch_scalar_rid_is_normalized_for_all_tp_rooms(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router._active_prefill_attempts = {}
        request = {"bootstrap_room": [201, 202], "rid": "batch"}

        router._set_prefill_attempt_rid(request, replace=False)
        self.assertEqual(request["rid"], ["batch_0", "batch_1"])
        router._activate_prefill_attempt(request, (201, 202))
        self.assertEqual(
            router._active_prefill_attempts,
            {201: "batch_0", 202: "batch_1"},
        )

    def test_prefill_attempt_duplicate_room_rejected_atomically(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router._active_prefill_attempts = {301: "existing"}
        request = {
            "bootstrap_room": [302, 301],
            "rid": ["new-rank-0", "new-rank-1"],
        }

        with self.assertRaisesRegex(RuntimeError, "already owned"):
            router._activate_prefill_attempt(request, (302, 301))
        self.assertEqual(router._active_prefill_attempts, {301: "existing"})

    def test_prefill_attempt_finally_style_cleanup_does_not_remove_new_owner(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router._active_prefill_attempts = {401: "new-owner"}

        router._deactivate_prefill_attempt(
            {"bootstrap_room": 401, "rid": "old-owner"}, (401,)
        )

        self.assertEqual(router._active_prefill_attempts, {401: "new-owner"})

    async def test_stale_monitor_snapshot_does_not_delete_current_owned_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.dynamic_prefill_domains = True
            router._active_prefill_attempts = {501: "current"}
            path = router._ready_path(501)
            path.write_bytes(
                orjson.dumps(
                    {
                        "rid": "current",
                        "prefill_domain": 0,
                        "ready_sequence": 8,
                    }
                )
            )
            router._p_ready_snapshot = {
                501: {
                    "_path": path,
                    "rid": "stale",
                    "prefill_domain": 0,
                    "ready_sequence": 7,
                }
            }

            async def dispatch():
                return 8

            self.assertEqual(
                await asyncio.wait_for(
                    router._dispatch_p_ready_in_order(8, 0, dispatch), timeout=1
                ),
                8,
            )
            self.assertTrue(path.exists())
            self.assertEqual(router._p_ready_snapshot[501]["rid"], "current")
            await router.close()

    async def test_d_admission_wait_does_not_hold_global_p_ready_fifo(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.early_claim_store = None
            router.max_prefill_inflight = 4
            router._prefill_admission = _PrefillAdmissionGate(limit=4)
            router._wait_until_prefill_accepted = AsyncMock()
            router._wait_until_prefill_scheduled = AsyncMock()
            router._wait_until_prefill_ready = AsyncMock(return_value=1024)
            router._p_ready_sequence = lambda rooms: 1
            reservation = DecodeReservation(
                reservation_id="reservation-1",
                url="http://d0",
                prompt_tokens=1024,
                admission_tokens=1536,
                request_count=1,
                rooms=(7,),
                created_at=0.0,
            )
            router._select_and_reserve_decode = AsyncMock(return_value=reservation)
            admission_started = asyncio.Event()
            release_admission = asyncio.Event()

            async def wait_for_admission(_reservation, _sequence=None):
                admission_started.set()
                await release_admission.wait()

            router._release_reservation_when_admitted = wait_for_admission

            class Response:
                status = 200

            class Session:
                async def post(self, *args, **kwargs):
                    return Response()

            task = asyncio.create_task(
                router._late_dispatch(
                    Session(),
                    {
                        "bootstrap_room": 7,
                        "sampling_params": {},
                        "return_logprob": False,
                    },
                    "http://p0",
                    "generate",
                    {},
                )
            )
            await asyncio.wait_for(admission_started.wait(), timeout=1)
            self.assertFalse(router._p_ready_fifo_waiters[0])
            release_admission.set()
            await asyncio.wait_for(task, timeout=1)
            await router.close()

    async def test_prefill_response_is_released_before_decode_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.early_claim_store = None
            router.max_prefill_inflight = 4
            router._prefill_admission = _PrefillAdmissionGate(4)
            router._wait_until_prefill_accepted = AsyncMock()
            router._wait_until_prefill_scheduled = AsyncMock()
            router._wait_until_prefill_ready = AsyncMock(return_value=1024)
            router._p_ready_sequence = lambda rooms: 1
            router._select_and_reserve_decode = AsyncMock(
                return_value=DecodeReservation(
                    reservation_id="response-lifecycle",
                    url="http://d0",
                    prompt_tokens=1024,
                    admission_tokens=1536,
                    request_count=1,
                    rooms=(7,),
                    created_at=0.0,
                )
            )
            router._release_reservation_when_admitted = AsyncMock()
            prefill_read = asyncio.Event()
            prefill_released = asyncio.Event()
            finish_decode = asyncio.Event()

            class PrefillResponse:
                status = 200

                async def read(self):
                    prefill_read.set()
                    return b"{}"

                def release(self):
                    prefill_released.set()

            class DecodeResponse:
                status = 200

            class Session:
                async def post(self, url, **kwargs):
                    if url == "http://d0/generate":
                        await finish_decode.wait()
                        return DecodeResponse()
                    return PrefillResponse()

            task = asyncio.create_task(
                router._late_dispatch(
                    Session(),
                    {
                        "bootstrap_room": 7,
                        "sampling_params": {},
                        "return_logprob": False,
                    },
                    "http://p0",
                    "generate",
                    {},
                )
            )
            await asyncio.wait_for(prefill_released.wait(), timeout=1)
            self.assertTrue(prefill_read.is_set())
            self.assertFalse(task.done())
            finish_decode.set()
            prefill_response, _ = await asyncio.wait_for(task, timeout=1)
            self.assertIsNone(prefill_response)

    async def test_prefill_read_failure_releases_p_and_cancels_d(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.early_claim_store = None
            router.max_prefill_inflight = 4
            router._prefill_admission = _PrefillAdmissionGate(4)
            router._wait_until_prefill_accepted = AsyncMock()
            router._wait_until_prefill_scheduled = AsyncMock()
            router._wait_until_prefill_ready = AsyncMock(return_value=1024)
            router._p_ready_sequence = lambda rooms: 1
            router._select_and_reserve_decode = AsyncMock(
                return_value=DecodeReservation(
                    reservation_id="response-read-failure",
                    url="http://d0",
                    prompt_tokens=1024,
                    admission_tokens=1536,
                    request_count=1,
                    rooms=(7,),
                    created_at=0.0,
                )
            )
            router._release_reservation_when_admitted = AsyncMock()
            prefill_released = asyncio.Event()
            decode_cancelled = asyncio.Event()

            class PrefillResponse:
                status = 200

                async def read(self):
                    raise ConnectionError("prefill body failed")

                def release(self):
                    prefill_released.set()

            class Session:
                async def post(self, url, **kwargs):
                    if url == "http://d0/generate":
                        try:
                            await asyncio.Future()
                        except asyncio.CancelledError:
                            decode_cancelled.set()
                            raise
                    return PrefillResponse()

            with self.assertRaisesRegex(ConnectionError, "prefill body failed"):
                await asyncio.wait_for(
                    router._late_dispatch(
                        Session(),
                        {"bootstrap_room": 7, "sampling_params": {}},
                        "http://p0",
                        "generate",
                        {},
                    ),
                    timeout=1,
                )
            self.assertTrue(prefill_released.is_set())
            self.assertTrue(decode_cancelled.is_set())

    async def test_dynamic_dispatch_aborts_then_reposts_on_numa_host_fallback(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router.dynamic_prefill_domains = True
            router._ensure_prefill_pressure_monitor = lambda: None
            router.numa_domains = True
            router.global_decode = True
            router.prefill_urls = ["http://p0", "http://p1"]
            router.prefill_bootstrap_ports = [100, 200]
            router._prefill_pending_tokens = [8000, 0]
            host_entry = {
                "state": HostStageState.HOST_READY.value,
                "arena_domain": 0,
            }

            def assign_recovery(_snapshot_id, domain):
                host_entry["recovery_domain"] = int(domain)
                return True

            router._d2p_host_ledger = types.SimpleNamespace(
                get=lambda _snapshot_id: dict(host_entry),
                assign_d2p_recovery_domain=assign_recovery,
                snapshot_entries=lambda: {"host": dict(host_entry)},
            )
            router.early_claim_store = AgenticEarlyClaimStore(
                str(Path(directory) / "claims")
            )
            router.max_prefill_inflight = 8
            router._prefill_admission = _PrefillAdmissionGate(8)
            parent = RequestGeneration("redirect", 0)
            router.early_claim_store.publish_route(
                parent,
                route="direct_ready",
                prefill_domain=0,
                snapshot_tokens=8000,
            )
            request = {
                "bootstrap_room": 99,
                # Chat Completions does not carry lifecycle metadata under
                # sampling_params.  The Router must recover it from the same
                # extra_key envelope that survives the HTTP request schema.
                "sampling_params": {},
                "custom_params": {
                    "agentic_request_id": "redirect",
                    "agentic_generation": 1,
                    "agentic_parent_generation": 0,
                    "agentic_prompt_token_count": 9000,
                },
            }
            # Protocol fixture, independent of the workload/client repository.
            stable = base64.urlsafe_b64encode(b"agentic-v1:redirect:g1").decode().rstrip("=")
            wire = {key: request["custom_params"][key] for key in (
                "agentic_request_id", "agentic_generation", "agentic_parent_generation")}
            payload = base64.urlsafe_b64encode(orjson.dumps(wire)).decode().rstrip("=")
            request["extra_key"] = f"agentic-v1e:{stable}:{payload}"

            accepted_calls = 0

            async def accepted(_rooms, _task, _route_task=None):
                nonlocal accepted_calls
                accepted_calls += 1
                if accepted_calls == 1:
                    router._prefill_pressure_at = time.monotonic()
                    router._prefill_pressure_domains = [
                        {
                            "domain": 0,
                            "hbm_capacity_tokens": 20_000,
                            "hbm_used_tokens": 0,
                        },
                        {
                            "domain": 1,
                            "hbm_capacity_tokens": 20_000,
                            "hbm_used_tokens": 15_000,
                        },
                    ]
                    router.early_claim_store.publish_route(
                        parent,
                        route="host_ready",
                        prefill_domain=-1,
                        arena_domain=0,
                        arena_numa_node=0,
                        snapshot_tokens=8000,
                    )

            async def scheduled(_rooms, _task, route_task=None):
                if route_task is not None:
                    await asyncio.wait_for(
                        asyncio.shield(route_task), timeout=1
                    )
                    router._raise_prefill_redirect(route_task)

            async def ready(_rooms, _task, _route_task=None):
                # Scheduled Slow H2D must proceed to P-ready while the
                # conservative shadow remains charged.  A causally newer
                # pressure sample retires the shadow in the background.
                self.assertGreater(sum(router._prefill_direct_pending_tokens), 0)
                router._prefill_pressure_sample_started_at = (
                    time.monotonic() + 0.001
                )
                return 9000

            router._wait_until_prefill_accepted = accepted
            router._wait_until_prefill_scheduled = scheduled
            router._wait_until_prefill_ready = ready
            router._p_ready_sequence = lambda _rooms: 0
            decode_reservation = DecodeReservation(
                reservation_id="redirect-d",
                url="http://d0",
                prompt_tokens=9000,
                admission_tokens=9512,
                request_count=1,
                rooms=(1,),
                created_at=0.0,
            )
            router._select_and_reserve_decode = AsyncMock(
                return_value=decode_reservation
            )
            router._release_reservation_when_admitted = AsyncMock()

            class Response:
                status = 200

                def release(self):
                    return None

            class Session:
                def __init__(self):
                    self.calls = []

                async def post(self, url, **kwargs):
                    self.calls.append(
                        (url, orjson.loads(orjson.dumps(kwargs.get("json"))))
                    )
                    return Response()

            session = Session()
            await asyncio.wait_for(
                router._late_dispatch(
                    session, request, "http://p0", "generate", {}
                ),
                timeout=2,
            )

            generate_calls = [call for call in session.calls if call[0].endswith("/generate")]
            self.assertEqual(
                [call[0] for call in generate_calls],
                ["http://p1/generate", "http://p0/generate", "http://d0/generate"],
            )
            self.assertIn(("http://p1/abort_request"), [call[0] for call in session.calls])
            self.assertNotEqual(
                generate_calls[0][1]["bootstrap_room"],
                generate_calls[1][1]["bootstrap_room"],
            )
            self.assertNotEqual(
                generate_calls[0][1]["rid"], generate_calls[1][1]["rid"]
            )
            self.assertEqual(router._prefill_admission.active, 0)
            await router.close()

    async def test_parent_direct_arrival_precedes_admission_and_gate_ends_at_accept(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory) / "ready")
            router.numa_domains = False
            router.dynamic_prefill_domains = False
            router.max_prefill_inflight = 1
            router._prefill_admission = _PrefillAdmissionGate(1)
            router.early_claim_store = AgenticEarlyClaimStore(
                str(Path(directory) / "claims")
            )
            await router._prefill_admission.acquire(parent_turn=False)

            router._wait_until_prefill_accepted = AsyncMock()
            router._wait_until_prefill_scheduled = AsyncMock()
            router._wait_until_prefill_ready = AsyncMock(return_value=4096)
            router._p_ready_sequence = lambda _rooms: 1
            router._select_and_reserve_decode = AsyncMock(
                return_value=DecodeReservation(
                    reservation_id="parent-admission",
                    url="http://d0",
                    prompt_tokens=4096,
                    admission_tokens=4608,
                    request_count=1,
                    rooms=(17,),
                    created_at=0.0,
                )
            )
            router._release_reservation_when_admitted = AsyncMock()

            prefill_started = asyncio.Event()
            finish_prefill = asyncio.Event()
            finish_decode = asyncio.Event()

            class Response:
                status = 200

                async def read(self):
                    return b"{}"

                def release(self):
                    return None

            class Session:
                async def post(self, url, **kwargs):
                    if url == "http://p0/generate":
                        prefill_started.set()
                        await finish_prefill.wait()
                    elif url == "http://d0/generate":
                        await finish_decode.wait()
                    return Response()

            parent = RequestGeneration("admission-agent", 0)
            request = {
                "bootstrap_room": 17,
                "input_ids": list(range(4096)),
                "sampling_params": {
                    "custom_params": {
                        "agentic_request_id": "admission-agent",
                        "agentic_generation": 1,
                        "agentic_parent_generation": 0,
                    }
                },
            }
            task = asyncio.create_task(
                router._late_dispatch(
                    Session(), request, "http://p0", "generate", {}
                )
            )
            await asyncio.sleep(0.02)
            arrival = router.early_claim_store.read_arrival(
                parent, not_before=0.0, max_age_seconds=5.0
            )
            self.assertIsNotNone(arrival)
            self.assertEqual(arrival["prompt_token_count"], 4096)

            await router._prefill_admission.release()
            await asyncio.wait_for(prefill_started.wait(), timeout=1)
            # HTTP acceptance, not the later P->D handoff, is the admission
            # boundary.  The exact parent+suffix workset lease independently
            # protects HBM until Prefill and delivery finish.
            await asyncio.sleep(0)
            self.assertEqual(router._prefill_admission.active, 0)
            blocked = asyncio.create_task(
                router._prefill_admission.acquire(parent_turn=True)
            )
            await asyncio.wait_for(blocked, timeout=1)
            self.assertFalse(task.done())

            finish_prefill.set()
            self.assertFalse(task.done())
            await router._prefill_admission.release()
            finish_decode.set()
            await asyncio.wait_for(task, timeout=1)
            self.assertEqual(router._prefill_admission.active, 0)

    async def test_same_generation_retries_join_one_detached_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            started = asyncio.Event()
            finish = asyncio.Event()
            calls = 0

            async def generate_once(request, prefill_server, endpoint):
                nonlocal calls
                calls += 1
                started.set()
                await finish.wait()
                return {"text": request["input_ids"]}, 200

            router._generate_once = generate_once
            request = {"input_ids": [1, 2, 3]}
            first = asyncio.create_task(
                router._generate_singleflight(
                    "agent:g2", request, "http://p0", "generate"
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            retry = asyncio.create_task(
                router._generate_singleflight(
                    "agent:g2", request, "http://p1", "generate"
                )
            )
            await asyncio.sleep(0)
            self.assertEqual(calls, 1)
            finish.set()
            self.assertEqual(await first, ({"text": [1, 2, 3]}, 200))
            self.assertEqual(await retry, ({"text": [1, 2, 3]}, 200))
            await asyncio.sleep(0)
            self.assertEqual(calls, 1)

    async def test_failed_generation_retry_uses_fresh_wire_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            attempts = []

            async def generate_once(request, _prefill_server, _endpoint):
                attempts.append((request["bootstrap_room"], request["rid"]))
                if len(attempts) == 1:
                    return {"error": "transport"}, 500
                return {"ok": True}, 200

            router._generate_once = generate_once
            request = {"bootstrap_room": 99, "rid": "external-rid"}

            self.assertEqual(
                await router._generate_singleflight(
                    "agent:g-failed", request, "http://p0", "generate"
                ),
                ({"error": "transport"}, 500),
            )
            # Let the done callback remove the failed producer. It must not
            # publish that 5xx into the completed-generation cache.
            await asyncio.sleep(0)
            self.assertEqual(
                await router._generate_singleflight(
                    "agent:g-failed", request, "http://p0", "generate"
                ),
                ({"ok": True}, 200),
            )
            self.assertEqual(len(attempts), 2)
            self.assertNotEqual(attempts[0], attempts[1])
            self.assertNotIn(99, {attempts[0][0], attempts[1][0]})
            self.assertNotIn(
                "external-rid", {attempts[0][1], attempts[1][1]}
            )

    async def test_abort_decode_attempt_cancels_every_tp_request_id(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        calls = []

        class Response:
            status = 200

            def release(self):
                return None

        class Session:
            async def post(self, url, json):
                calls.append((url, json))
                return Response()

        self.assertTrue(
            await router._abort_decode_attempt(
                Session(),
                "http://d0",
                {"rid": ["rank-0", "rank-1"]},
            )
        )
        self.assertEqual(
            calls,
            [
                ("http://d0/abort_request", {"rid": "rank-0"}),
                ("http://d0/abort_request", {"rid": "rank-1"}),
            ],
        )

    async def test_client_cancellation_does_not_cancel_generation_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            started = asyncio.Event()
            finish = asyncio.Event()
            calls = 0

            async def generate_once(_request, _prefill_server, _endpoint):
                nonlocal calls
                calls += 1
                started.set()
                await finish.wait()
                return {"ok": True}, 200

            router._generate_once = generate_once
            client = asyncio.create_task(
                router._generate_singleflight(
                    "agent:g3", {}, "http://p0", "generate"
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            client.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await client
            finish.set()
            retry = await router._generate_singleflight(
                "agent:g3", {}, "http://p1", "generate"
            )
            self.assertEqual(retry, ({"ok": True}, 200))
            self.assertEqual(calls, 1)

    async def test_accepted_marker_releases_router_admission_before_p_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            room = 23
            router._accepted_path(room).write_bytes(
                orjson.dumps({"rid": "request-b"})
            )
            unfinished = asyncio.create_task(asyncio.sleep(10))
            try:
                await router._wait_until_prefill_accepted((room,), unfinished)
            finally:
                unfinished.cancel()
            self.assertFalse(router._accepted_path(room).exists())

    async def test_scheduled_marker_starts_processing_timeout_after_queue_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            room = 24
            router._scheduled_path(room).write_bytes(
                orjson.dumps({"rid": "request-c"})
            )
            unfinished = asyncio.create_task(asyncio.sleep(10))
            try:
                await router._wait_until_prefill_scheduled((room,), unfinished)
            finally:
                unfinished.cancel()
            self.assertFalse(router._scheduled_path(room).exists())

    async def test_prefill_admission_is_bounded_and_fifo_across_turn_types(self):
        gate = _PrefillAdmissionGate(limit=1)
        await gate.acquire(parent_turn=False)
        fresh_new = asyncio.create_task(gate.acquire(parent_turn=False))
        parent = asyncio.create_task(gate.acquire(parent_turn=True))
        await asyncio.sleep(0)

        await gate.release()
        await asyncio.wait_for(fresh_new, timeout=1)
        self.assertFalse(parent.done())
        self.assertEqual(gate.active, 1)

        await gate.release()
        await asyncio.wait_for(parent, timeout=1)
        await gate.release()
        self.assertEqual(gate.active, 0)

    def test_prefill_admission_next_waiter_is_fifo(self):
        gate = _PrefillAdmissionGate(limit=1)
        now = 100.0
        aged_new = _PrefillAdmissionWaiter(False, now - 11, 0)
        parent = _PrefillAdmissionWaiter(True, now, 1)
        gate._waiters = [parent, aged_new]

        self.assertIs(gate._next_waiter(now), aged_new)

    async def test_prefill_admission_capacity_is_independent_per_p(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router._prefill_admissions = [
            _PrefillAdmissionGate(limit=1),
            _PrefillAdmissionGate(limit=1),
        ]

        p0 = router._prefill_admission_for_domain(0)
        p1 = router._prefill_admission_for_domain(1)
        await p0.acquire(parent_turn=False)
        p0_waiter = asyncio.create_task(p0.acquire(parent_turn=True))
        await asyncio.sleep(0)

        # Saturating P0 does not consume P1's independent capacity.
        await asyncio.wait_for(p1.acquire(parent_turn=False), timeout=1)
        self.assertFalse(p0_waiter.done())
        self.assertEqual((p0.active, p1.active), (1, 1))

        await p0.release()
        await asyncio.wait_for(p0_waiter, timeout=1)
        await p0.release()
        await p1.release()

    async def test_atomic_reservation_spreads_simultaneous_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            equal_loads = [
                DecodeLoad("http://d0", 10_000, 100_000, 10, 0, 0, 0, 100),
                DecodeLoad("http://d1", 10_000, 100_000, 10, 0, 0, 0, 100),
            ]

            async def loads(_session, *, force=False):
                return equal_loads

            router._all_decode_loads = loads
            first = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (1,), 20_000
            )
            second = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (2,), 20_000
            )
            self.assertEqual(first.url, "http://d0")
            self.assertEqual(second.url, "http://d1")

    async def test_admitted_reservation_remains_charged_until_new_load_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            reservation = DecodeReservation(
                reservation_id="admitted-1",
                url="http://d0",
                prompt_tokens=10_000,
                admission_tokens=12_000,
                request_count=1,
                rooms=(91,),
                created_at=1.0,
            )
            router._reservations[reservation.reservation_id] = reservation
            router._load_cache_at = 10.0
            router._load_cache_sample_started_at = 10.0
            router._admitted_reservation_at[reservation.reservation_id] = 11.0

            self.assertEqual(router._reserved_for("http://d0"), (10_000, 12_000, 1))

            router._load_cache_at = 12.0
            router._load_cache_sample_started_at = 12.0
            router._prune_accounted_reservations()
            self.assertEqual(router._reserved_for("http://d0"), (0, 0, 0))
            self.assertNotIn(reservation.reservation_id, router._reservations)

    async def test_load_sample_started_before_admission_cannot_clear_credit(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        old_poll_started = asyncio.Event()
        release_old_poll = asyncio.Event()

        async def fetch(_session, url):
            old_poll_started.set()
            await release_old_poll.wait()
            return DecodeLoad(url, 10_000, 100_000, 10, 0, 0, 0, 100)

        router._fetch_decode_load = fetch
        old_poll = asyncio.create_task(router._refresh_decode_loads(None))
        await asyncio.wait_for(old_poll_started.wait(), timeout=1)

        reservation = DecodeReservation(
            reservation_id="causal-credit",
            url="http://d0",
            prompt_tokens=10_000,
            admission_tokens=12_000,
            request_count=1,
            rooms=(92,),
            created_at=time.monotonic(),
        )
        router._reservations[reservation.reservation_id] = reservation
        admitted_at = time.monotonic()
        router._admitted_reservation_at[reservation.reservation_id] = admitted_at

        release_old_poll.set()
        await old_poll
        self.assertLess(router._load_cache_sample_started_at, admitted_at)
        router._prune_accounted_reservations()
        self.assertEqual(router._reserved_for("http://d0"), (10_000, 12_000, 1))

        # A poll whose sampling begins after admission is causally allowed to
        # observe it and therefore retire the Router-side credit.
        await router._refresh_decode_loads(None)
        self.assertGreaterEqual(router._load_cache_sample_started_at, admitted_at)
        router._prune_accounted_reservations()
        self.assertNotIn(reservation.reservation_id, router._reservations)

    async def test_older_load_refresh_cannot_overwrite_newer_epoch(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        old_started = asyncio.Event()
        release_old = asyncio.Event()
        calls = 0

        async def fetch(_session, url):
            nonlocal calls
            calls += 1
            call = calls
            if call <= len(router.decode_urls):
                old_started.set()
                await release_old.wait()
                used = 80_000
            else:
                used = 20_000
            return DecodeLoad(url, used, 100_000, 10, 0, 0, 0, 100)

        router._fetch_decode_load = fetch
        old_poll = asyncio.create_task(router._refresh_decode_loads(None))
        await asyncio.wait_for(old_started.wait(), timeout=1)
        await asyncio.sleep(0)
        new_loads = await router._refresh_decode_loads(None)
        newer_epoch = router._load_cache_sample_started_at
        self.assertTrue(all(load.used_tokens == 20_000 for load in new_loads))

        release_old.set()
        old_result = await old_poll
        self.assertEqual(router._load_cache_sample_started_at, newer_epoch)
        self.assertTrue(all(load.used_tokens == 20_000 for load in old_result))
        self.assertTrue(
            all(load.used_tokens == 20_000 for load in router._load_cache)
        )

    async def test_initial_decode_load_refresh_is_singleflight(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        refresh_started = asyncio.Event()
        release_refresh = asyncio.Event()
        refreshes = 0
        loads = [
            DecodeLoad("http://d0", 10_000, 100_000, 10, 0, 0, 0, 100),
            DecodeLoad("http://d1", 10_000, 100_000, 10, 0, 0, 0, 100),
        ]

        async def refresh(_session):
            nonlocal refreshes
            refreshes += 1
            refresh_started.set()
            await release_refresh.wait()
            router._load_cache = loads
            router._load_cache_at = time.monotonic()
            return loads

        router._refresh_decode_loads = refresh
        waiters = [
            asyncio.create_task(router._all_decode_loads(None))
            for _ in range(64)
        ]
        await asyncio.wait_for(refresh_started.wait(), timeout=1)
        await asyncio.sleep(0)
        self.assertEqual(refreshes, 1)

        waiters[0].cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiters[0]
        release_refresh.set()
        results = await asyncio.gather(*waiters[1:])
        self.assertEqual(refreshes, 1)
        self.assertTrue(all(result is loads for result in results))

    async def test_failed_d_poll_cannot_clear_that_d_reservation_credit(self):
        router = self.make_router(Path("/dev/shm/test-ready"))

        async def initial_fetch(_session, url):
            return DecodeLoad(url, 10_000, 100_000, 10, 0, 0, 0, 100)

        router._fetch_decode_load = initial_fetch
        await router._refresh_decode_loads(None)

        reservation = DecodeReservation(
            reservation_id="partial-refresh-credit",
            url="http://d0",
            prompt_tokens=10_000,
            admission_tokens=12_000,
            request_count=1,
            rooms=(93,),
            created_at=time.monotonic(),
        )
        router._reservations[reservation.reservation_id] = reservation
        router._admitted_reservation_at[reservation.reservation_id] = time.monotonic()

        async def partial_fetch(_session, url):
            if url == "http://d0":
                raise RuntimeError("d0 unavailable")
            return DecodeLoad(url, 20_000, 100_000, 10, 0, 0, 0, 100)

        router._fetch_decode_load = partial_fetch
        await router._refresh_decode_loads(None)
        router._prune_accounted_reservations()
        self.assertIn(reservation.reservation_id, router._reservations)
        self.assertEqual(router._reserved_for("http://d0"), (10_000, 12_000, 1))

        router._fetch_decode_load = initial_fetch
        await router._refresh_decode_loads(None)
        router._prune_accounted_reservations()
        self.assertNotIn(reservation.reservation_id, router._reservations)

    async def test_all_failed_load_poll_does_not_refresh_stale_cache_ttl(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        stale = [
            DecodeLoad(url, 90_000, 100_000, 10, 0, 0, 0, 100)
            for url in router.decode_urls
        ]
        router._load_cache = stale
        router._last_loads = {load.url: load for load in stale}
        router._load_cache_at = 123.0
        router._load_cache_sample_started_at = 122.0

        async def failed_fetch(_session, _url):
            raise RuntimeError("control plane unavailable")

        router._fetch_decode_load = failed_fetch
        result = await router._refresh_decode_loads(None)

        self.assertIs(result, stale)
        self.assertEqual(router._load_cache_at, 123.0)
        self.assertEqual(router._load_cache_sample_started_at, 122.0)

    async def test_legacy_load_network_wait_runs_outside_router_loop(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        loop_thread = threading.get_ident()
        worker_threads = []
        original = late_binding_router_module._sync_json_get

        def fetch(_url, _timeout):
            worker_threads.append(threading.get_ident())
            return [
                {
                    "num_reqs": 7,
                    "num_waiting_reqs": 1,
                    "num_physical_used_tokens": 1234,
                    "num_running_kv_tokens": 1000,
                    "max_total_num_tokens": 100_000,
                    "max_running_requests": 128,
                }
            ]

        late_binding_router_module._sync_json_get = fetch
        try:
            load = await router._fetch_decode_load_legacy(None, "http://d0")
        finally:
            late_binding_router_module._sync_json_get = original

        self.assertEqual(load.running, 6)
        self.assertEqual(load.used_tokens, 1234)
        self.assertEqual(len(worker_threads), 1)
        self.assertNotEqual(worker_threads[0], loop_thread)

    async def test_least_running_dominates_kv_pressure_after_capacity_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            loads_now = [
                DecodeLoad("http://d0", 90_000, 100_000, 5, 0, 0, 0, 100),
                DecodeLoad("http://d1", 40_000, 100_000, 30, 0, 0, 0, 100),
            ]

            async def loads(_session, *, force=False):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (3,), 5_000
            )
            self.assertEqual(selected.url, "http://d0")

    async def test_capacity_filter_dominates_least_running(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            loads_now = [
                DecodeLoad("http://d0", 95_000, 100_000, 5, 0, 0, 0, 100),
                DecodeLoad("http://d1", 40_000, 100_000, 30, 0, 0, 0, 100),
            ]

            async def loads(_session, *, force=False):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (31,), 5_000
            )
            self.assertEqual(selected.url, "http://d1")

    async def test_target_kv_margin_buffers_p_ready_work(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.target_decode_kv_fraction = 0.90
            loads_now = [
                DecodeLoad("http://d0", 85_000, 100_000, 5, 0, 0, 0, 100),
                DecodeLoad("http://d1", 60_000, 100_000, 30, 0, 0, 0, 100),
            ]

            async def loads(_session, *, force=False):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (32,), 5_000
            )
            # d0 has fewer requests, but admitting this P-ready snapshot would
            # cross the 90% operating target.  Keep it buffered on P and use
            # the D that still has growth/egress headroom.
            self.assertEqual(selected.url, "http://d1")

    async def test_p_ready_admission_fills_available_d_capacity_continuously(self):
        """A prior P->D submission need not complete before the next one starts."""

        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.target_decode_kv_fraction = 0.90
            loads_now = [
                DecodeLoad("http://d0", 40_000, 100_000, 20, 0, 0, 0, 100)
            ]

            async def loads(_session, *, force=False):
                return loads_now

            router._all_decode_loads = loads
            first = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (321,), 5_000
            )
            second = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (322,), 5_000
            )

            self.assertEqual(first.url, "http://d0")
            self.assertEqual(second.url, "http://d0")
            self.assertEqual(len(router._reservations), 2)

    async def test_running_kv_work_can_outweigh_a_smaller_request_count(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            loads_now = [
                DecodeLoad(
                    "http://d0", 50_000, 100_000, 10, 0, 0, 0, 100,
                    running_kv_tokens=80_000,
                ),
                DecodeLoad(
                    "http://d1", 30_000, 100_000, 12, 0, 0, 0, 100,
                    running_kv_tokens=24_000,
                ),
            ]

            async def loads(_session, *, force=False):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (311,), 5_000
            )
            self.assertEqual(selected.url, "http://d1")

    async def test_transfer_gets_an_extra_interference_penalty(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            loads_now = [
                DecodeLoad(
                    "http://d0", 30_000, 100_000, 10, 3, 0, 3, 100,
                    running_kv_tokens=30_000,
                    transfer_tokens=6_000,
                ),
                DecodeLoad(
                    "http://d1", 30_000, 100_000, 13, 0, 0, 0, 100,
                    running_kv_tokens=36_000,
                ),
            ]

            async def loads(_session, *, force=False):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (312,), 5_000
            )
            self.assertEqual(selected.url, "http://d1")

    async def test_handoff_queue_counts_as_projected_decode_work(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            loads_now = [
                DecodeLoad("http://d0", 20_000, 100_000, 10, 4, 4, 0, 100),
                DecodeLoad("http://d1", 20_000, 100_000, 10, 0, 0, 0, 100),
            ]

            async def loads(_session, *, force=False):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (32,), 5_000
            )
            self.assertEqual(selected.url, "http://d1")

    async def test_direct_admission_skips_d_with_full_transfer_window(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.max_direct_transfer_inflight = 4
            loads_now = [
                # Lower compute load, but all four Direct slots are owned.
                DecodeLoad("http://d0", 10_000, 100_000, 1, 0, 0, 4, 100),
                DecodeLoad("http://d1", 30_000, 100_000, 10, 0, 0, 1, 100),
            ]

            async def loads(_session, *, force=False):
                return loads_now

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (3201,), 5_000
            )
            self.assertEqual(selected.url, "http://d1")

    async def test_waits_for_real_decode_capacity_instead_of_preallocating(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            attempts = 0

            async def loads(_session, *, force=False):
                nonlocal attempts
                attempts += 1
                used = 99_000 if attempts == 1 else 80_000
                return [
                    DecodeLoad("http://d0", used, 100_000, 50, 0, 0, 0, 100),
                    DecodeLoad("http://d1", 99_000, 100_000, 50, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (4,), 5_000
            )
            self.assertEqual(selected.url, "http://d0")
            self.assertGreaterEqual(attempts, 2)

    async def test_p2d_host_claim_can_restore_to_global_least_work_d(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.prefill_urls = ["http://p0", "http://p1"]
            router.decode_urls = [
                "http://d0",
                "http://d1",
                "http://d2",
                "http://d3",
            ]
            router.numa_domains = True
            router.global_decode = True
            router.p2d_host_spill_delay = 0.0
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            snapshot_id = p2d_snapshot_id(401)
            attempts = 0

            async def loads(_session, *, force=False):
                nonlocal attempts
                attempts += 1
                entry = ledger.get(snapshot_id)
                if entry is not None and entry["state"] == HostStageState.OFFERED.value:
                    claimed = ledger.claim(snapshot_id, "p2d-p:test")
                    self.assertIsNotNone(claimed)
                    self.assertTrue(
                        ledger.publish_grants(
                            snapshot_id,
                            "p2d-p:test",
                            [{"kind": "shared_host_extent"}],
                        )
                    )
                    self.assertTrue(ledger.ack_chunk(snapshot_id, "p2d-p:test", 0))
                    self.assertTrue(
                        ledger.mark_host_ready(snapshot_id, "p2d-p:test", 1)
                    )
                used = 99_000 if attempts == 1 else 20_000
                return [
                    DecodeLoad("http://d0", used, 100_000, 20, 0, 0, 0, 100),
                    DecodeLoad("http://d1", used, 100_000, 25, 0, 0, 0, 100),
                    # Remote NUMA is deliberately less loaded.  Durable Host
                    # ownership must not prevent global late binding.
                    DecodeLoad("http://d2", used, 100_000, 1, 0, 0, 0, 100),
                    DecodeLoad("http://d3", used, 100_000, 2, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (401,), 5_000, domain=0
            )
            self.assertEqual(selected.url, "http://d2")
            self.assertEqual(selected.p2d_host_snapshot_id, snapshot_id)
            self.assertEqual(selected.prefill_domain, 0)

    async def test_p2d_cancel_live_writer_enters_aborting_until_fence(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.global_decode = True
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            snapshot_id = p2d_snapshot_id(424)
            owner = "p2d-p:test"
            ledger.offer(
                {
                    "snapshot_id": snapshot_id,
                    "token_count": 5_000,
                    "prefill_domain": 0,
                    "control_offer": True,
                }
            )
            self.assertIsNotNone(ledger.claim(snapshot_id, owner))
            self.assertTrue(
                ledger.transition(
                    snapshot_id, HostStageState.HOST_WRITING, owner=owner
                )
            )

            async def loads(_session, *, force=False):
                return [
                    DecodeLoad("http://d0", 99_000, 100_000, 50, 0, 0, 0, 100),
                    DecodeLoad("http://d1", 99_000, 100_000, 50, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selector = asyncio.create_task(
                router._select_and_reserve_decode(
                    None, {"max_tokens": 1000}, (424,), 5_000, domain=0
                )
            )
            for _ in range(100):
                if router._reservations:
                    break
                await asyncio.sleep(0.001)
            selector.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await selector

            self.assertEqual(
                ledger.get(snapshot_id)["state"], HostStageState.ABORTING.value
            )
            self.assertEqual(router._reservations, {})
            # Only the physical producer may publish FAILED after its CUDA
            # fence.  The Router cancellation above must not skip this step.
            self.assertTrue(
                ledger.transition(snapshot_id, HostStageState.FAILED, owner=owner)
            )
            self.assertEqual(
                ledger.get(snapshot_id)["state"], HostStageState.FAILED.value
            )

    def test_p2d_dispatch_abort_never_rolls_back_live_h2d(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            snapshot_id = p2d_snapshot_id(425)
            owner = "p2d-p:test"
            ledger.offer(
                {
                    "snapshot_id": snapshot_id,
                    "token_count": 5_000,
                    "control_offer": True,
                }
            )
            self.assertIsNotNone(ledger.claim(snapshot_id, owner))
            self.assertTrue(
                ledger.publish_grants(
                    snapshot_id, owner, [{"kind": "shared_host_extent"}]
                )
            )
            self.assertTrue(ledger.ack_chunk(snapshot_id, owner, 0))
            self.assertTrue(ledger.mark_host_ready(snapshot_id, owner, 1))
            self.assertTrue(
                ledger.begin_host_load_rank(
                    snapshot_id, owner, tp_rank=0, tp_size=1
                )
            )
            self.assertEqual(
                ledger.get(snapshot_id)["state"], HostStageState.H2D_LOADING.value
            )

            router._abort_unsubmitted_p2d(snapshot_id, "router_dispatch_failed")
            self.assertEqual(
                ledger.get(snapshot_id)["state"], HostStageState.H2D_LOADING.value
            )

    def test_p2d_dispatch_abort_can_close_unloaded_host_ready(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            snapshot_id = p2d_snapshot_id(426)
            owner = "p2d-p:test"
            ledger.offer(
                {
                    "snapshot_id": snapshot_id,
                    "token_count": 5_000,
                    "control_offer": True,
                }
            )
            self.assertIsNotNone(ledger.claim(snapshot_id, owner))
            self.assertTrue(
                ledger.publish_grants(
                    snapshot_id, owner, [{"kind": "shared_host_extent"}]
                )
            )
            self.assertTrue(ledger.ack_chunk(snapshot_id, owner, 0))
            self.assertTrue(ledger.mark_host_ready(snapshot_id, owner, 1))

            router._abort_unsubmitted_p2d(snapshot_id, "router_dispatch_failed")
            self.assertEqual(
                ledger.get(snapshot_id)["state"], HostStageState.FAILED.value
            )
    async def test_global_host_restore_does_not_wait_for_missing_local_load(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.prefill_urls = ["http://p0", "http://p1"]
            router.decode_urls = [
                "http://d0",
                "http://d1",
                "http://d2",
                "http://d3",
            ]
            router.numa_domains = True
            router.global_decode = True
            router.p2d_host_spill_delay = 0.0
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            snapshot_id = p2d_snapshot_id(451)
            self.assertEqual(
                router._publish_p2d_host_offer(
                    snapshot_id, (451,), 5_000, domain=0, source="test"
                ),
                snapshot_id,
            )
            self.assertIsNotNone(ledger.claim(snapshot_id, "p2d-p:test"))
            self.assertTrue(
                ledger.publish_grants(
                    snapshot_id,
                    "p2d-p:test",
                    [{"kind": "shared_host_extent"}],
                )
            )
            self.assertTrue(ledger.ack_chunk(snapshot_id, "p2d-p:test", 0))
            self.assertTrue(ledger.mark_host_ready(snapshot_id, "p2d-p:test", 1))

            remote = [
                DecodeLoad("http://d2", 20_000, 100_000, 10, 0, 0, 0, 100),
                DecodeLoad("http://d3", 25_000, 100_000, 11, 0, 0, 0, 100),
            ]

            async def loads(_session, *, force=False):
                self.assertFalse(force)
                return remote

            router._all_decode_loads = loads
            result = await asyncio.wait_for(
                router._select_and_reserve_decode(
                    None, {"max_tokens": 1000}, (451,), 5_000, domain=0
                ),
                timeout=1,
            )
            self.assertEqual(result.url, "http://d2")
            self.assertEqual(result.p2d_host_snapshot_id, snapshot_id)
            self.assertFalse(router._selection_lock.locked())

    async def test_only_blocked_fifo_head_can_publish_p2d_host_offer(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.global_decode = True
            router.numa_domains = False
            router.p2d_host_spill_delay = 0.0
            router._p2d_host_offered_snapshots = set()
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            router._all_decode_loads = AsyncMock(
                return_value=[
                    DecodeLoad("http://d0", 99_000, 100_000, 50, 0, 0, 0, 100),
                    DecodeLoad("http://d1", 99_000, 100_000, 50, 0, 0, 0, 100),
                ]
            )
            first = asyncio.create_task(
                router._select_and_reserve_decode(
                    None, {"max_tokens": 1000}, (406,), 5_000, domain=0
                )
            )
            deadline = time.monotonic() + 1
            while (
                ledger.get(p2d_snapshot_id(406)) is None
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.001)

            self.assertEqual(
                ledger.get(p2d_snapshot_id(406))["state"],
                HostStageState.OFFERED.value,
            )
            # Capacity pressure is local to the admission being selected.  It
            # must not publish storage ownership for a later completed room.
            self.assertIsNone(ledger.get(p2d_snapshot_id(407)))
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            await router.close()

    async def test_p2d_unclaimed_offer_is_cancelled_when_d_frees(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.global_decode = True
            router.p2d_host_spill_delay = 0.0
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            attempts = 0

            async def loads(_session, *, force=False):
                nonlocal attempts
                attempts += 1
                used = 99_000 if attempts == 1 else 20_000
                return [
                    DecodeLoad("http://d0", used, 100_000, 10, 0, 0, 0, 100),
                    DecodeLoad("http://d1", used, 100_000, 11, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (402,), 5_000, domain=0
            )
            self.assertIsNone(selected.p2d_host_snapshot_id)
            self.assertEqual(
                ledger.get(p2d_snapshot_id(402))["state"],
                HostStageState.REJECTED.value,
            )

    async def test_p2d_host_spill_rechecks_a_causally_fresh_global_load(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = True
            router.global_decode = True
            router.p2d_host_spill_delay = 0.0
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            router._p2d_host_offered_snapshots = set()

            full = [
                DecodeLoad("http://d0", 99_000, 100_000, 50, 0, 0, 0, 100),
                DecodeLoad("http://d1", 99_000, 100_000, 50, 0, 0, 0, 100),
            ]
            refreshed = [
                DecodeLoad("http://d0", 99_000, 100_000, 50, 0, 0, 0, 100),
                DecodeLoad("http://d1", 20_000, 100_000, 10, 0, 0, 0, 100),
            ]
            router._load_cache = full
            router._last_loads = {load.url: load for load in full}
            router._load_sample_started_at_by_url = {
                load.url: 0.0 for load in full
            }
            loads_now = full

            async def loads(_session, *, force=False):
                return loads_now

            async def observe(_session, *, urls, not_before):
                nonlocal loads_now
                loads_now = refreshed
                router._load_cache = refreshed
                router._load_sample_started_at_by_url = {
                    url: not_before for url in urls
                }
                return True

            router._all_decode_loads = loads
            router._observe_decode_load_after = observe
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (403,), 5_000, domain=0
            )

            self.assertEqual(selected.url, "http://d1")
            self.assertIsNone(selected.p2d_host_snapshot_id)
            self.assertIsNone(ledger.get(p2d_snapshot_id(403)))
            self.assertEqual(len(router._reservations), 1)
            self.assertFalse(next(iter(router._reservations.values())).draining)

    async def test_failed_fresh_load_check_still_allows_p2d_host_progress(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = True
            router.global_decode = True
            router.p2d_host_spill_delay = 0.0
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            router._p2d_host_offered_snapshots = set()
            full = [
                DecodeLoad("http://d0", 99_000, 100_000, 50, 0, 0, 0, 100),
                DecodeLoad("http://d1", 99_000, 100_000, 50, 0, 0, 0, 100),
            ]
            router._load_cache = full
            router._last_loads = {load.url: load for load in full}
            router._load_sample_started_at_by_url = {
                load.url: 0.0 for load in full
            }
            router._all_decode_loads = AsyncMock(return_value=full)
            router._observe_decode_load_after = AsyncMock(return_value=False)

            selector = asyncio.create_task(
                router._select_and_reserve_decode(
                    None, {"max_tokens": 1000}, (404,), 5_000, domain=0
                )
            )
            deadline = time.monotonic() + 1
            while (
                ledger.get(p2d_snapshot_id(404)) is None
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.001)

            self.assertEqual(
                ledger.get(p2d_snapshot_id(404))["state"],
                HostStageState.OFFERED.value,
            )
            selector.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await selector

    async def test_p2d_ledger_arbitration_never_holds_d_selection_lock(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.global_decode = True
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            router._p2d_host_offered_snapshots = set()
            snapshot_id = p2d_snapshot_id(499)
            ledger.offer(
                {
                    "snapshot_id": snapshot_id,
                    "token_count": 1_000,
                    "prefill_domain": 0,
                    "control_offer": True,
                }
            )
            original_get = ledger.get
            original_reject = ledger.reject_unclaimed_offer

            def checked_get(snapshot):
                self.assertFalse(router._selection_lock.locked())
                return original_get(snapshot)

            def checked_reject(snapshot, *, reason):
                self.assertFalse(router._selection_lock.locked())
                return original_reject(snapshot, reason=reason)

            ledger.get = checked_get
            ledger.reject_unclaimed_offer = checked_reject
            router._all_decode_loads = AsyncMock(
                return_value=[
                    DecodeLoad(
                        "http://d0", 0, 100_000, 0, 0, 0, 0, 128
                    )
                ]
            )
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 512}, (499,), 1_000, domain=0
            )
            self.assertEqual(selected.url, "http://d0")
            self.assertEqual(
                original_get(snapshot_id)["state"],
                HostStageState.REJECTED.value,
            )

    async def test_cancel_between_host_finalize_and_credit_commit_rolls_back(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.global_decode = True
            router.p2d_host_ledger = None
            router._all_decode_loads = AsyncMock(
                return_value=[
                    DecodeLoad(
                        "http://d0", 0, 100_000, 0, 0, 0, 0, 128
                    )
                ]
            )
            finalize_started = threading.Event()
            release_finalize = threading.Event()

            def finalize(reservation, **_kwargs):
                finalize_started.set()
                release_finalize.wait(timeout=1)
                return reservation

            router._finalize_p2d_route = finalize
            selector = asyncio.create_task(
                router._select_and_reserve_decode(
                    None, {"max_tokens": 512}, (501,), 1_000, domain=0
                )
            )
            while not finalize_started.is_set():
                await asyncio.sleep(0.001)
            await router._selection_lock.acquire()
            release_finalize.set()
            await asyncio.sleep(0.01)
            selector.cancel()
            router._selection_lock.release()
            with self.assertRaises(asyncio.CancelledError):
                await selector
            self.assertFalse(router._reservations)

    async def test_cancel_waits_for_late_host_offer_before_abort(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.global_decode = True
            router.p2d_host_spill_delay = 0.0
            ledger = SharedHostStagingLedger(str(Path(directory) / "p2d.json"))
            router.p2d_host_ledger = ledger
            router._p2d_host_offered_snapshots = set()
            router._all_decode_loads = AsyncMock(
                return_value=[
                    DecodeLoad(
                        "http://d0", 99_000, 100_000, 30, 0, 0, 0, 128
                    )
                ]
            )
            offer_started = threading.Event()
            release_offer = threading.Event()
            original_publish = router._publish_p2d_host_offer

            def delayed_publish(*args, **kwargs):
                offer_started.set()
                release_offer.wait(timeout=1)
                return original_publish(*args, **kwargs)

            router._publish_p2d_host_offer = delayed_publish
            selector = asyncio.create_task(
                router._select_and_reserve_decode(
                    None, {"max_tokens": 512}, (502,), 5_000, domain=0
                )
            )
            while not offer_started.is_set():
                await asyncio.sleep(0.001)
            selector.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(selector.done())
            release_offer.set()
            with self.assertRaises(asyncio.CancelledError):
                await selector
            entry = ledger.get(p2d_snapshot_id(502))
            self.assertIsNotNone(entry)
            self.assertNotEqual(entry["state"], HostStageState.OFFERED.value)
            self.assertFalse(router._reservations)

    def test_p2d_failed_host_owner_cannot_fall_back_to_direct(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.p2d_host_ledger = SharedHostStagingLedger(
                str(Path(directory) / "p2d.json")
            )
            snapshot_id = p2d_snapshot_id(420)
            router.p2d_host_ledger.offer({"snapshot_id": snapshot_id})
            router.p2d_host_ledger.transition(
                snapshot_id, HostStageState.FAILED, reason="d2h_failed"
            )
            reservation = DecodeReservation(
                "r-failed", "http://d0", 1024, 2048, 1, (420,), time.monotonic()
            )

            with self.assertRaisesRegex(RuntimeError, "exclusive ownership"):
                router._finalize_p2d_route(
                    reservation,
                    snapshot_id=snapshot_id,
                    state=HostStageState.FAILED.value,
                    domain=0,
                )

    async def test_p2d_failed_feasible_selector_leaves_no_capacity_credit(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as directory:
            router = self.make_router(Path(directory))
            router.numa_domains = False
            router.global_decode = True
            router.p2d_host_ledger = SharedHostStagingLedger(
                str(Path(directory) / "p2d.json")
            )
            snapshot_id = p2d_snapshot_id(423)
            router.p2d_host_ledger.offer({"snapshot_id": snapshot_id})
            router.p2d_host_ledger.transition(
                snapshot_id, HostStageState.FAILED, reason="d2h_failed"
            )

            async def loads(_session, *, force=False):
                return [
                    DecodeLoad("http://d0", 10_000, 100_000, 8, 0, 0, 0, 100),
                    DecodeLoad("http://d1", 20_000, 100_000, 9, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            with self.assertRaisesRegex(RuntimeError, "exclusive ownership"):
                await router._select_and_reserve_decode(
                    None, {"max_tokens": 1000}, (423,), 5_000, domain=0
                )
            self.assertEqual(router._reservations, {})

    def test_p2d_consumed_stale_attempt_cannot_submit_again(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router.p2d_host_ledger = types.SimpleNamespace(get=lambda _snapshot: {})
        reservation = DecodeReservation(
            "r-consumed", "http://d0", 1024, 2048, 1, (421,), time.monotonic()
        )

        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            router._finalize_p2d_route(
                reservation,
                snapshot_id=p2d_snapshot_id(421),
                state=HostStageState.CONSUMED.value,
                domain=0,
            )

    def test_p2d_aborting_host_owner_waits_instead_of_direct(self):
        router = self.make_router(Path("/dev/shm/test-ready"))
        router.p2d_host_ledger = types.SimpleNamespace(get=lambda _snapshot: {})
        reservation = DecodeReservation(
            "r-aborting", "http://d0", 1024, 2048, 1, (422,), time.monotonic()
        )

        self.assertIsNone(
            router._finalize_p2d_route(
                reservation,
                snapshot_id=p2d_snapshot_id(422),
                state=HostStageState.ABORTING.value,
                domain=0,
            )
        )

    async def test_ordered_head_immediately_reserves_future_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            attempts = 0

            async def loads(_session, *, force=False):
                nonlocal attempts
                attempts += 1
                # Initially neither D can fit 31k.  Once the ordered credit is
                # installed on d0, Decode drain makes enough room there.
                d0_used = 95_000 if attempts < 3 else 60_000
                return [
                    DecodeLoad("http://d0", d0_used, 100_000, 50, 0, 0, 0, 100),
                    DecodeLoad("http://d1", 95_000, 100_000, 50, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (5,), 30_000
            )
            self.assertEqual(selected.url, "http://d0")
            self.assertTrue(selected.draining)
            self.assertGreaterEqual(attempts, 3)

    async def test_draining_credit_reselects_when_old_d_exceeds_target(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            router.target_decode_kv_fraction = 0.90
            attempts = 0

            async def loads(_session, *, force=False):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    # Both are infeasible, and the lower projected fill makes
                    # d0 the initial draining hint.
                    used = (92_000, 95_000)
                else:
                    # The old d0 draining hint still hard-fits 31k, but its
                    # projected 91% exceeds the configured 90% target.  d1 is
                    # the only feasible destination.
                    used = (60_000, 10_000)
                return [
                    DecodeLoad("http://d0", used[0], 100_000, 50, 0, 0, 0, 100),
                    DecodeLoad("http://d1", used[1], 100_000, 10, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (501,), 30_000
            )

            self.assertEqual(selected.url, "http://d1")
            self.assertFalse(selected.draining)
            self.assertEqual(len(router._reservations), 1)
            self.assertIs(router._reservations[selected.reservation_id], selected)

    async def test_draining_credit_reselects_to_least_work_feasible_d(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self.make_router(Path(directory))
            attempts = 0

            async def loads(_session, *, force=False):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    used = (99_000, 99_000)
                else:
                    # Both workers now fit this 6k admission, but d1 has much
                    # less work.  The old all-full d0 hint must not pin it.
                    used = (60_000, 10_000)
                return [
                    DecodeLoad("http://d0", used[0], 100_000, 50, 0, 0, 0, 100),
                    DecodeLoad("http://d1", used[1], 100_000, 10, 0, 0, 0, 100),
                ]

            router._all_decode_loads = loads
            selected = await router._select_and_reserve_decode(
                None, {"max_tokens": 1000}, (502,), 5_000
            )

            self.assertEqual(selected.url, "http://d1")
            self.assertEqual(len(router._reservations), 1)
            self.assertEqual(
                next(iter(router._reservations.values())).reservation_id,
                selected.reservation_id,
            )


if __name__ == "__main__":
    unittest.main()
