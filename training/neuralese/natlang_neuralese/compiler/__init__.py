"""The Natlang Compiler: unified slot-to-weights compilation & experience engine (`plans/NATLANG_COMPILER.md`)."""
from .collapse import (
    COLLAPSE_SCHEMA,
    CollapsedMemoryBundle,
    CollapsedRelease,
    collapse_mode_p_memory,
    collapse_to_lora_factors,
    provenance_key,
)
from .header_encoder import HeaderEncoder, HeaderEncoderConfig, default_lfm3_tap_layers
from .unified_head import CompiledSiteOutput, UnifiedCompilerHead, compose_cores

__all__ = [
    "COLLAPSE_SCHEMA",
    "CollapsedMemoryBundle",
    "CollapsedRelease",
    "CompiledSiteOutput",
    "HeaderEncoder",
    "HeaderEncoderConfig",
    "UnifiedCompilerHead",
    "collapse_mode_p_memory",
    "collapse_to_lora_factors",
    "compose_cores",
    "default_lfm3_tap_layers",
    "provenance_key",
]
