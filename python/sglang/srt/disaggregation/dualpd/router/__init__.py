"""Engine-owned agentic routers.

Service modules are deliberately lazy: the TCP router registers FastAPI routes
on import, so importing this package must not activate either router variant.
"""
