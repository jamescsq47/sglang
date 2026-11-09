"""Router protocol bindings; no worker construction or physical ownership.

Ordinary maintained imports must fail closed. Only the
historical private file-package entry may use its selected old native overlay.
"""
from importlib import import_module


def _selected_module(leaf, legacy_leaf):
    target = "sglang.srt.disaggregation." + leaf
    try:
        return import_module(target)
    except ModuleNotFoundError as exc:
        if (
            __package__ != "dualpd_engine_late_binding_router"
            or exc.name != target
        ):
            raise
        return import_module("sglang.srt.disaggregation." + legacy_leaf)


_ledger = _selected_module("agentic_host_ledger", "agentic_host_staging")
HostStageState = _ledger.HostStageState
SharedHostStagingLedger = _ledger.SharedHostStagingLedger

_protocol = _selected_module("p2d_host_protocol", "p2d_host_staging")
P2D_CUSTOM_PREFILL_DOMAIN = _protocol.P2D_CUSTOM_PREFILL_DOMAIN
P2D_CUSTOM_SNAPSHOT_ID = _protocol.P2D_CUSTOM_SNAPSHOT_ID
p2d_snapshot_id = _protocol.p2d_snapshot_id
