"""Monitoring is opt-in and does not participate in ownership decisions."""
from sglang.srt.disaggregation.agentic_control_rpc import ControlRPCClient
from sglang.srt.disaggregation.agentic_control_server import create_server
from sglang.srt.disaggregation.agentic_tp_events import TPEventServer


def test_tp_stats_rpc_uses_existing_authenticated_control_service():
    events = TPEventServer("stats", "secret", max_entries=24)
    server = create_server("stats", "secret", ("127.0.0.1", 0), tp_server=events)
    client = ControlRPCClient(server.address, run_id="stats", token="secret")
    try:
        assert client.call("system", "describe")["run_id"] == "stats"
        stats = client.call("system", "tp_stats")
        assert stats["entry_count"] == 0
        assert stats["max_entries"] == 24
        assert stats["groups"] == {}
    finally:
        client.close()
        server.close()
        events.close()
