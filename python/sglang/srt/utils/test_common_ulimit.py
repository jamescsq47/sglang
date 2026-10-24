import resource

from sglang.srt.utils import common


def test_explicit_stack_limit_can_lower_sglang_default(monkeypatch):
    calls = []

    def getrlimit(kind):
        if kind == resource.RLIMIT_NOFILE:
            return 65535, 65535
        assert kind == resource.RLIMIT_STACK
        return 65535 * 1024, resource.RLIM_INFINITY

    monkeypatch.setenv("SGLANG_STACK_SOFT_LIMIT_KIB", "8192")
    monkeypatch.setattr(common.resource, "getrlimit", getrlimit)
    monkeypatch.setattr(
        common.resource,
        "setrlimit",
        lambda kind, limits: calls.append((kind, limits)),
    )

    common.set_ulimit()

    assert calls == [
        (resource.RLIMIT_STACK, (8192 * 1024, resource.RLIM_INFINITY))
    ]


def test_default_stack_limit_never_lowers_existing_limit(monkeypatch):
    calls = []

    def getrlimit(kind):
        if kind == resource.RLIMIT_NOFILE:
            return 65535, 65535
        assert kind == resource.RLIMIT_STACK
        return 128 * 1024 * 1024, resource.RLIM_INFINITY

    monkeypatch.delenv("SGLANG_STACK_SOFT_LIMIT_KIB", raising=False)
    monkeypatch.setattr(common.resource, "getrlimit", getrlimit)
    monkeypatch.setattr(
        common.resource,
        "setrlimit",
        lambda kind, limits: calls.append((kind, limits)),
    )

    common.set_ulimit()

    assert calls == []
