"""The Natlang Compiler (`plans/NATLANG_COMPILER.md`).

``compiled_site``: a site compiles to the program tuner's stack (header native state + xs cores; Mode P), served as
stored by every engine. ``unified_head``: hindsight-kb's MemoryKVBank + AdapterReader over Z (Mode W, dynamic Stage-C
HKB research on the hindsight-kb decoder; imports hindsight-kb, so it is not imported here).
"""
from .collapse import COLLAPSE_SCHEMA, provenance_key
from .compiled_site import (COMPILED_SITE_SCHEMA, CompiledSite, CompiledSiteConfig, CompiledSiteHead,
                            compile_release, site_header, site_loss)
from .header_encoder import HeaderEncoder, HeaderEncoderConfig, default_lfm3_tap_layers

__all__ = [
    "COLLAPSE_SCHEMA", "COMPILED_SITE_SCHEMA", "CompiledSite", "CompiledSiteConfig", "CompiledSiteHead",
    "HeaderEncoder", "HeaderEncoderConfig", "compile_release", "default_lfm3_tap_layers", "provenance_key",
    "site_header", "site_loss",
]
