"""LiquidAI LFM3 modeling code, vendored byte-identical from the pinned Hub revision (owner, 2026-10-28).

Transformers 5.18 has no ``lfm3``. These three files are the remote code of ``LiquidAI/LFM3-1.7B`` at revision
``40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd``; their SHA-256 are verified before import, and they are registered
with the Transformers Auto classes so ``AutoConfig``/``AutoModelForCausalLM`` load local LFM3 checkpoints with
``trust_remote_code=False``. They are the exact parity reference of the native port (``model/hf_port.Lfm3PortBackbone``).
License: LFM Open License v1.0 (``LICENSE``, from the same revision).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPOSITORY = "LiquidAI/LFM3-1.7B"
REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
SOURCES = {
    "modeling_lfm3.py": "e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab",
    "configuration_lfm3.py": "176cced35657aedfea7d7002221360582d1c05fcaca882c21508bc3a73b282e5",
    "lfm3_conv.py": "80d4a985382a22774212759248a6b4c737bfbf9c2a47b3dd7f797dda748e4bb4",
    "LICENSE": "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c",
}
# Weights of the pinned revision (not vendored; read from the local snapshot).
WEIGHTS_SHA256 = "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"

_registered = False


def verify_sources() -> dict:
    """The recorded SHA-256 of each vendored file, after checking the files on disk against them."""
    root = Path(__file__).resolve().parent
    for name, expected in SOURCES.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"vendored LFM3 source {name} has sha256 {actual}, expected {expected}")
    return dict(SOURCES)


def register():
    """Verify the vendored sources and register ``lfm3`` with the Transformers Auto classes (idempotent)."""
    global _registered
    if _registered:
        return
    verify_sources()
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    from .configuration_lfm3 import Lfm3Config
    from .modeling_lfm3 import Lfm3ForCausalLM, Lfm3Model

    AutoConfig.register("lfm3", Lfm3Config, exist_ok=True)
    AutoModel.register(Lfm3Config, Lfm3Model, exist_ok=True)
    AutoModelForCausalLM.register(Lfm3Config, Lfm3ForCausalLM, exist_ok=True)
    _registered = True


def config_class():
    register()
    from .configuration_lfm3 import Lfm3Config
    return Lfm3Config


def causal_lm_class():
    register()
    from .modeling_lfm3 import Lfm3ForCausalLM
    return Lfm3ForCausalLM
