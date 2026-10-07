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
def device():
    """Where model-backed tests run: the GPU when it has room, else the CPU. NATLANG_TEST_DEVICE overrides it, and the
    ledger's NATLANG_CUDA_MEMORY_GB caps CUDA allocations."""
    from natlang_neuralese.devices import auto_device, cap_cuda_memory

    chosen = os.environ.get("NATLANG_TEST_DEVICE") or auto_device()
    cap_cuda_memory(chosen, float(os.environ.get("NATLANG_CUDA_MEMORY_GB") or 0) or None)
    return chosen


@pytest.fixture(autouse=True)
def _tensors_on_model_device(request):
    """Tests on the model device (the shared backbone, or `device` itself) create their tensors there (torch's
    default-device context)."""
    if "device" not in request.fixturenames:
        yield
        return
    with torch.device(request.getfixturevalue("device")):
        yield


class _TokenizerOnDevice:
    """The backbone's tokenizer, returning ``return_tensors="pt"`` encodings on the model's device."""

    def __init__(self, tokenizer, device):
        self._tokenizer, self._device = tokenizer, device

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        encoded = self._tokenizer(*args, **kwargs)
        return encoded.to(self._device) if kwargs.get("return_tensors") == "pt" else encoded


@pytest.fixture(scope="session")
def loaded(device):
    from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone, load_backbone, resolve_base

    if not Path(resolve_base(None)).exists():
        pytest.skip("LFM2.5-350M is not in the local HF cache")
    torch.manual_seed(0)
    model, tokenizer = load_backbone(dtype=torch.float32, device=device)
    if device != "cpu":
        tokenizer = _TokenizerOnDevice(tokenizer, device)
    backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer))
    return model, tokenizer, backbone


@pytest.fixture(scope="session")
def heads(loaded, device):
    from natlang_neuralese.model.heads import PortHeads

    torch.manual_seed(1)
    return PortHeads(loaded[2], cutoff=6, max_length=8).to(device).eval()
