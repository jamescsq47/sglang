"""Native disaggregation plus the isolated DualPD extension.

The public ``disaggregation.agentic_*`` module names remain stable so model
workers, routers and historical launchers share one module identity. Extension
sources live under ``dualpd/``; Python resolves them through this package path.
No extension module, CUDA state, or worker is initialized by this import.
"""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent / "dualpd"))
