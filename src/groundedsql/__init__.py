"""GroundedSQL — ReAct SQL agent with grounding verification, evaluated on BIRD."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("groundedsql")
except PackageNotFoundError:
    __version__ = "0.0.0"
