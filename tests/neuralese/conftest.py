import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "neuralese"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


_chosen: list[str] = []


@pytest.fixture(scope="session")
def device():
    """Where model-backed tests run: the GPU when it has room, else the CPU. NATLANG_TEST_DEVICE overrides it, and the
    ledger's NATLANG_CUDA_MEMORY_GB caps CUDA allocations."""
    from natlang_neuralese.devices import auto_device, cap_cuda_memory

    chosen = os.environ.get("NATLANG_TEST_DEVICE") or auto_device()
    cap_cuda_memory(chosen, float(os.environ.get("NATLANG_CUDA_MEMORY_GB") or 0) or None)
    _chosen[:] = [chosen]
    return chosen


def _device_context(device):
    """Torch's default-device context: tensors and modules created inside land on ``device`` (none: unchanged)."""
    import contextlib

    return torch.device(device) if device else contextlib.nullcontext()


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    """A test on the model device (one that uses `device`, directly or through the shared backbone) gets its fixtures
    built there too, module-scoped heads and engines included."""
    device = None
    if fixturedef.argname != "device" and "device" in request.fixturenames:
        device = request.getfixturevalue("device") if fixturedef.scope == "function" else (_chosen or [None])[0]
    with _device_context(device):
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    with _device_context(pyfuncitem.funcargs.get("device")):
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
