from .dialect import DIALECT, StoredBlock, block_id
from .lfm2_port import ControlTokens, PortBackbone, PortCache, load_backbone

__all__ = [
    "DIALECT", "StoredBlock", "block_id",
    "ContentProjection", "FeedbackProjection", "InterfaceNorm", "PortHeads", "StopHead",
    "ControlTokens", "PortBackbone", "PortCache", "load_backbone",
]


def __getattr__(name):
    # The native positional-state runtime shares lfm2_port's backbone interface,
    # but does not construct the retired PortHeads adapter. Keep old trainer
    # imports source-compatible without importing that module on every native
    # model import.
    if name in {"ContentProjection", "FeedbackProjection", "InterfaceNorm", "PortHeads", "StopHead"}:
        from . import heads
        return getattr(heads, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
