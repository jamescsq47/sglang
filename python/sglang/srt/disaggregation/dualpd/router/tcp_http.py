"""TCP backend requests and generation retry lifetime on the concrete Router."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid

import aiohttp
from fastapi.responses import ORJSONResponse, StreamingResponse


class TCPHTTPMixin:
    """Method-only component; no constructor, independent queue or callback registration."""

    def _backend_timeout(self):
        return aiohttp.ClientTimeout(
            total=self._generation_timeout,
            connect=self.timeout,
            sock_connect=self.timeout,
            sock_read=self._generation_timeout,
        )

    @staticmethod
    async def _settle_backend_task(task, label):
        """Cancel and consume one backend request without masking D success.

        A non-logprob response is wholly owned by Decode.  Prefill's HTTP
        response carries no user data and may be lost after its readiness
        callback and ownership handoff have already succeeded.  Waiting for
        that unused response before Decode used to strand the client for the
        full generation timeout and made the outer HTTP retry replay the same
        immutable generation.  Always observe the task, but treat it as
        cleanup once Decode has supplied the authoritative response.
        """

        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            logging.warning("%s backend cleanup failed: %r", label, error)

    @staticmethod
    def _generation_key(request):
        custom = request.get("custom_params") or {}
        request_id = custom.get("agentic_request_id")
        generation = custom.get("agentic_generation")
        if request_id is None or generation is None:
            return None
        return str(request_id), int(generation)

    def _prune_generation_tasks(self):
        now = asyncio.get_running_loop().time()
        expired = [
            key
            for key, completed_at in self._generation_completed_at.items()
            if now - completed_at >= self._generation_retry_ttl
        ]
        for key in expired:
            self._generation_completed_at.pop(key, None)
            self._generation_tasks.pop(key, None)

        overflow = len(self._generation_tasks) - self._generation_retry_cache_max
        if overflow <= 0:
            return
        oldest = sorted(
            self._generation_completed_at.items(), key=lambda item: item[1]
        )
        for key, _completed_at in oldest[:overflow]:
            self._generation_completed_at.pop(key, None)
            self._generation_tasks.pop(key, None)

    def _record_generation_done(self, key, task):
        if self._generation_tasks.get(key) is task:
            self._generation_completed_at[key] = asyncio.get_running_loop().time()

    def _release_duplicate_prefill_selection(self, request, purl):
        # MiniLB calls select_pair() before generate(), so even an idempotent
        # retry arrives with a fresh provisional P charge.  Consume and
        # release that shadow charge without touching either physical worker.
        prefill_tokens = self._materialize_prefill_reservation(purl, request)
        self._release(purl, prefill_tokens=prefill_tokens)

    async def generate(self, request, purl, durl, endpoint):
        key = self._generation_key(request)
        if key is None:
            return await self._generate_once(request, purl, durl, endpoint)

        self._prune_generation_tasks()
        task = self._generation_tasks.get(key)
        if task is not None:
            self._release_duplicate_prefill_selection(request, purl)
        else:
            task = asyncio.create_task(
                self._generate_once(request, purl, durl, endpoint),
                name=f"dualpd-generation-{key[0]}-{key[1]}",
            )
            self._generation_tasks[key] = task
            task.add_done_callback(
                lambda completed, generation_key=key: self._record_generation_done(
                    generation_key, completed
                )
            )

        # A caller disconnect cancels only this waiter.  P/D ownership and
        # transport continue to their normal fence, and a retry joins `task`.
        return await asyncio.shield(task)

    async def _generate_once(self, request, purl, durl, endpoint):
        del durl
        prefill_tokens = self._materialize_prefill_reservation(purl, request)
        prefill_request = copy.deepcopy(request)
        self._annotate_prefill(prefill_request, purl)
        selected_d = None
        decode_reservation_id = None
        p_released = False
        prefill_task = None
        decode_task = None
        try:
            async with aiohttp.ClientSession(
                timeout=self._backend_timeout()
            ) as session:
                async def run_prefill():
                    nonlocal p_released
                    try:
                        return await session.post(
                            f"{purl}/{endpoint}", json=prefill_request
                        )
                    finally:
                        if not p_released:
                            self._release(purl, prefill_tokens=prefill_tokens)
                            p_released = True

                prefill_task = asyncio.create_task(run_prefill())
                ready = await self._wait_prefill_ready(prefill_request)
                decode_reservation_id = f"{self._room(prefill_request)}:{uuid.uuid4().hex}"
                selected_d = await self._select_decode(
                    ready.get("required_tokens", 0), decode_reservation_id
                )
                decode_request = copy.deepcopy(prefill_request)
                self._annotate_decode(
                    decode_request, selected_d, decode_reservation_id
                )
                decode_task = asyncio.create_task(
                    session.post(f"{selected_d}/{endpoint}", json=decode_request)
                )
                if "return_logprob" in request:
                    prefill_response = await prefill_task
                    decode_response = await decode_task
                    prefill_json = await prefill_response.json()
                    ret_json = await decode_response.json()
                    if (
                        "meta_info" in ret_json
                        and "input_token_logprobs" in ret_json["meta_info"]
                    ):
                        ret_json["meta_info"]["input_token_logprobs"] = (
                            prefill_json["meta_info"]["input_token_logprobs"]
                            + ret_json["meta_info"]["input_token_logprobs"]
                        )
                else:
                    # Decode owns the user-visible result.  Do not let an
                    # already-unneeded P response hold this request open.
                    decode_response = await decode_task
                    ret_json = await decode_response.json()
                return ORJSONResponse(
                    content=ret_json, status_code=decode_response.status
                )
        finally:
            await self._settle_backend_task(prefill_task, "Prefill")
            await self._settle_backend_task(decode_task, "Decode")
            if not p_released:
                self._release(purl, prefill_tokens=prefill_tokens)
            if selected_d is not None:
                self._inflight[selected_d.rstrip("/")] -= 1
            if decode_reservation_id is not None:
                self._release_decode_reservation(decode_reservation_id)

    async def generate_stream(self, request, purl, durl, endpoint="generate"):
        del durl
        prefill_tokens = self._materialize_prefill_reservation(purl, request)
        prefill_request = copy.deepcopy(request)
        self._annotate_prefill(prefill_request, purl)

        async def stream_results():
            selected_d = None
            decode_reservation_id = None
            p_released = False
            prefill_task = None
            decode_task = None
            try:
                async with aiohttp.ClientSession(
                    timeout=self._backend_timeout()
                ) as session:
                    async def run_prefill():
                        nonlocal p_released
                        try:
                            return await session.post(
                                f"{purl}/{endpoint}", json=prefill_request
                            )
                        finally:
                            if not p_released:
                                self._release(
                                    purl, prefill_tokens=prefill_tokens
                                )
                                p_released = True

                    prefill_task = asyncio.create_task(run_prefill())
                    ready = await self._wait_prefill_ready(prefill_request)
                    decode_reservation_id = (
                        f"{self._room(prefill_request)}:{uuid.uuid4().hex}"
                    )
                    selected_d = await self._select_decode(
                        ready.get("required_tokens", 0), decode_reservation_id
                    )
                    decode_request = copy.deepcopy(prefill_request)
                    self._annotate_decode(
                        decode_request, selected_d, decode_reservation_id
                    )
                    decode_task = asyncio.create_task(
                        session.post(f"{selected_d}/{endpoint}", json=decode_request)
                    )
                    if request.get("return_logprob", False):
                        prefill_response = await prefill_task
                        decode_response = await decode_task
                        prefill_chunks = [chunk async for chunk in prefill_response.content]
                        first = json.loads(
                            prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                        )
                        async for chunk in decode_response.content:
                            decoded = chunk.decode("utf-8")
                            if decoded.startswith("data:") and "[DONE]" not in decoded:
                                value = json.loads(decoded[5:].strip("\n"))
                                value["meta_info"]["input_token_logprobs"] = (
                                    first["meta_info"]["input_token_logprobs"]
                                    + value["meta_info"]["input_token_logprobs"]
                                )
                                yield b"data: " + json.dumps(value).encode() + b"\n\n"
                            else:
                                yield chunk
                    else:
                        decode_response = await decode_task
                        async for chunk in decode_response.content.iter_chunked(65536):
                            yield chunk
            finally:
                await self._settle_backend_task(prefill_task, "Prefill")
                await self._settle_backend_task(decode_task, "Decode")
                if not p_released:
                    self._release(purl, prefill_tokens=prefill_tokens)
                if selected_d is not None:
                    self._inflight[selected_d.rstrip("/")] -= 1
                if decode_reservation_id is not None:
                    self._release_decode_reservation(decode_reservation_id)

        return StreamingResponse(stream_results(), media_type="text/event-stream")
