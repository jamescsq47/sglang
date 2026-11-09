"""CLI entry point for the experimental late-binding PD router."""

import logging
import sys

import sglang_router.mini_lb as mini_lb_module
import uvicorn
from sglang_router.launch_router import parse_router_args

if __package__:
    from sglang.srt.disaggregation.dualpd.router.late_binding_router import LateBindingMiniLoadBalancer
else:
    # A historical overlay may not contain this new package. Load only the
    # Router source, leaving all sglang imports bound to the selected overlay.
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "dualpd_engine_late_binding_router", Path(__file__).with_name("late_binding_router.py"),
        submodule_search_locations=[str(Path(__file__).parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    LateBindingMiniLoadBalancer = module.LateBindingMiniLoadBalancer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_router_args(sys.argv[1:])
    args.mini_lb = True
    router = LateBindingMiniLoadBalancer(args)
    # The FastAPI handlers in sglang_router.mini_lb resolve this module-global.
    mini_lb_module.lb = router
    # FastAPI 0.1xx exposes lifecycle registration on its APIRouter rather
    # than directly on the application object.
    mini_lb_module.app.router.add_event_handler("shutdown", router.close)
    if router.enable_trace:
        mini_lb_module.process_tracing_init(router.otlp_traces_endpoint, "sglang")
        mini_lb_module.trace_set_thread_info("Mini lb")
    # This Router is both an ASGI server and a high-concurrency aiohttp client.
    # Under c512, uvloop can reuse an fd that it still associates with an
    # inbound ASGI transport while aiohappyeyeballs is opening a backend
    # connection.  The resulting cross-wired socket first raises
    # ``fd is used by transport`` and can then parse our outbound POST as an
    # inbound response.  Keeping only the Router on the standard asyncio loop
    # avoids that unsafe mixed server/client fd lifecycle; P and D workers keep
    # their normal event-loop configuration.
    uvicorn.run(
        mini_lb_module.app,
        host=router.host,
        port=router.port,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
