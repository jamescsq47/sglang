"""TCP load sampling and causal shadow reservations; one concrete state owner."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

import aiohttp


@dataclass(slots=True)
class _DecodeReservation:
    reservation_id: str
    tokens: int
    expires_at: float
    materialized_after_started_epoch: int | None = None


class TCPCapacityMixin:
    """Method-only component; physical TP authorities still grant memory."""

    def _least(self, values):
        start = self._tie
        self._tie += 1
        return min(
            enumerate(values),
            key=lambda item: (
                (
                    (
                        self._load[item[1].rstrip("/")][0]
                        + self._prefill_reserved_tokens.get(
                            item[1].rstrip("/"), 0
                        )
                    )
                    / self._load[item[1].rstrip("/")][1]
                    if self._load[item[1].rstrip("/")][1] > 0
                    else 0.0
                ),
                self._load[item[1].rstrip("/")][2],
                self._inflight[item[1].rstrip("/")],
                (item[0] - start) % len(values),
            ),
        )[0]

    async def _sample(self, session, url):
        endpoint = url.rstrip("/")
        sample_epoch = 0
        if endpoint in self._load_started_epoch:
            self._load_started_epoch[endpoint] += 1
            sample_epoch = self._load_started_epoch[endpoint]
        try:
            async with session.get(
                f"{url.rstrip('/')}/get_load",
                timeout=aiohttp.ClientTimeout(total=1.0),
            ) as response:
                response.raise_for_status()
                rows = await response.json()
            if isinstance(rows, dict):
                rows = [rows]
            used = sum(
                int(row.get("num_physical_used_tokens", row.get("num_tokens", 0)))
                for row in rows
            )
            capacity = sum(
                max(
                    0,
                    int(row.get("max_total_num_tokens", 0))
                    - (
                        self._decode_growth_reserve_tokens
                        if endpoint in self._decode_reservations
                        else 0
                    ),
                )
                for row in rows
            )
            waiting = sum(int(row.get("num_waiting_reqs", 0)) for row in rows)
            self._load[endpoint] = (used, capacity, waiting)
            if endpoint in self._load_epoch:
                self._load_epoch[endpoint] = sample_epoch
                self._release_sampled_decode_reservations(endpoint)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError):
            # Physical TP reservation is still authoritative.  A stale load
            # sample may affect balancing but can never grant memory.
            return

    async def _poll(self):
        async with aiohttp.ClientSession() as session:
            while True:
                await asyncio.gather(
                    *(self._sample(session, url) for url in self._inflight),
                    return_exceptions=True,
                )
                await asyncio.sleep(0.2)

    async def start_polling(self):
        if self._poll_task is None:
            self._poll_task = asyncio.create_task(self._poll())

    @staticmethod
    def _prompt_tokens(request):
        custom = request.get("custom_params") or {}
        value = custom.get("agentic_prompt_token_count")
        if value is not None:
            return max(1, int(value))
        values = request.get("input_ids")
        if isinstance(values, list) and (not values or isinstance(values[0], int)):
            return max(1, len(values))
        return None

    def _materialize_prefill_reservation(self, purl, request):
        endpoint = purl.rstrip("/")
        if self._pending_prefill_reservations[endpoint] <= 0:
            raise RuntimeError("missing provisional Prefill reservation")
        self._pending_prefill_reservations[endpoint] -= 1
        exact = self._prompt_tokens(request) or self._default_prefill_reservation
        self._prefill_reserved_tokens[endpoint] += (
            (exact - self._default_prefill_reservation) * self._tp_size
        )
        return exact

    async def _select_decode(self, required_tokens, reservation_id=None):
        required_tokens = max(0, int(required_tokens))
        reservation_id = str(reservation_id or uuid.uuid4().hex)
        if reservation_id in self._decode_reservation_index:
            raise RuntimeError("duplicate Decode reservation identity")
        deadline = asyncio.get_running_loop().time() + self.timeout
        while True:
            self._prune_decode_reservations()
            feasible = []
            for index, url in enumerate(self.decode_urls):
                endpoint = url.rstrip("/")
                used, capacity, waiting = self._load[endpoint]
                projected = used + self._decode_reserved_tokens[endpoint]
                if capacity <= 0 or capacity - projected >= required_tokens:
                    feasible.append(
                        (index, url, projected, capacity, waiting)
                    )
            if feasible:
                start = self._tie
                self._tie += 1
                _index, selected, *_ = min(
                    feasible,
                    key=lambda item: (
                        item[2] / item[3] if item[3] > 0 else 0.0,
                        item[4],
                        self._inflight[item[1].rstrip("/")],
                        (item[0] - start) % len(self.decode_urls),
                    ),
                )
                endpoint = selected.rstrip("/")
                self._inflight[endpoint] += 1
                self._decode_reserved_tokens[endpoint] += required_tokens
                reservation = _DecodeReservation(
                    reservation_id=reservation_id,
                    tokens=required_tokens,
                    expires_at=(
                        time.monotonic() + self._decode_reservation_seconds
                    ),
                )
                self._decode_reservations[endpoint][reservation_id] = reservation
                self._decode_reservation_index[reservation_id] = endpoint
                return selected
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("no Decode TP group can reserve the complete workset")
            await asyncio.sleep(0.02)

    def _prune_decode_reservations(self, now=None):
        now = time.monotonic() if now is None else float(now)
        for endpoint, reservations in self._decode_reservations.items():
            expired = [
                reservation_id
                for reservation_id, reservation in reservations.items()
                if reservation.expires_at <= now
            ]
            for reservation_id in expired:
                self._release_decode_reservation(reservation_id)
            if self._decode_reserved_tokens[endpoint] < 0:
                raise RuntimeError("Decode token reservation underflow")

    def _release_decode_reservation(self, reservation_id):
        reservation_id = str(reservation_id)
        endpoint = self._decode_reservation_index.pop(reservation_id, None)
        if endpoint is None:
            return False
        reservation = self._decode_reservations[endpoint].pop(reservation_id)
        self._decode_reserved_tokens[endpoint] -= reservation.tokens
        if self._decode_reserved_tokens[endpoint] < 0:
            raise RuntimeError("Decode token reservation underflow")
        return True

    async def publish_decode_materialized(self, payload):
        reservation_id = str(payload.get("reservation_id", ""))
        target_group = str(payload.get("target_group", ""))
        if not reservation_id or not target_group:
            raise ValueError("Decode materialized callback omitted reservation identity")
        endpoint = self._decode_reservation_index.get(reservation_id)
        if endpoint is None:
            # Idempotent retry after a newer physical sample already consumed
            # the shadow reservation.
            return
        if self._groups.get(endpoint) != target_group:
            raise ValueError("Decode PREPARED target differs from Router reservation")
        reservation = self._decode_reservations[endpoint][reservation_id]
        if reservation.materialized_after_started_epoch is None:
            # Capture requests that had already started, not merely samples
            # that had completed.  A GET issued before DMA_DONE but returning
            # afterward is not allowed to consume this shadow reservation.
            reservation.materialized_after_started_epoch = (
                self._load_started_epoch[endpoint]
            )

    def _release_sampled_decode_reservations(self, endpoint):
        epoch = self._load_epoch[endpoint]
        ready = [
            reservation_id
            for reservation_id, reservation in self._decode_reservations[
                endpoint
            ].items()
            if reservation.materialized_after_started_epoch is not None
            and reservation.materialized_after_started_epoch < epoch
        ]
        for reservation_id in ready:
            self._release_decode_reservation(reservation_id)

    def _release(self, purl, durl=None, *, prefill_tokens=0):
        endpoint = purl.rstrip("/")
        self._inflight[endpoint] -= 1
        if prefill_tokens:
            self._prefill_reserved_tokens[endpoint] -= (
                int(prefill_tokens) * self._tp_size
            )
            if self._prefill_reserved_tokens[endpoint] < 0:
                raise RuntimeError("Prefill token reservation underflow")
        if durl is not None:
            self._inflight[durl.rstrip("/")] -= 1
