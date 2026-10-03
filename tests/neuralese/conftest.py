import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "neuralese"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="session")
def loaded():
    from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone, load_backbone, resolve_base

    if not Path(resolve_base(None)).exists():
        pytest.skip("LFM2.5-350M is not in the local HF cache")
    torch.manual_seed(0)
    model, tokenizer = load_backbone(dtype=torch.float32, device="cpu")
    backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer))
    return model, tokenizer, backbone


@pytest.fixture(scope="session")
def heads(loaded):
    from natlang_neuralese.model.heads import PortHeads

    torch.manual_seed(1)
    return PortHeads(loaded[2], cutoff=6, max_length=8).eval()
