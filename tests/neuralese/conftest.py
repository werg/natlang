import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# The long-lived suite has checkpoint threads before its first CUDA compile.
# On DGX its default Inductor subprocess/fork pool stalled on inherited locks
# (full 8b60f49b9e); the identical attention test compiled and passed in 6.44 s
# with one worker. Keep this test-process setting explicit and overridable;
# production trainers and their kernel/gradient arithmetic are unchanged.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
# Trainer tests never append to the machine's checkpoint-write log the memory guard reads.
os.environ.setdefault("NATLANG_CHECKPOINT_WRITES", os.path.join(os.environ.get("TMPDIR", "/tmp"), "natlang-test-checkpoint-writes.jsonl"))
# Trainer tests never mirror their checkpoints to the machine's external disk (tests of the offload enable it).
os.environ.setdefault("NATLANG_CHECKPOINT_OFFLOAD", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "neuralese"))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu_heavy: needs several GB of CUDA memory; scripts/check-main.sh runs these in "
                            "a pytest process of their own after the rest, so their peak is not added to the suite's")


@pytest.fixture(autouse=True)
def _isolate_stop_signal_handlers():
    """Put back SIGTERM/SIGINT handlers after each test. A trainer's ``main`` installs a StopSignal and leaves it in
    place when it returns, so after the first in-process trainer test the pytest process swallowed SIGTERM: a ledger
    preemption had to SIGKILL it after its grace (full check 724ece2f, 2026-10-13 00:09), and a SIGTERM landing in a
    later trainer test stopped that run before its first update (checkpointed-on-signal at step 0)."""
    import signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    yield
    for sig, handler in saved.items():
        if signal.getsignal(sig) is not handler:
            signal.signal(sig, handler)


@pytest.fixture(autouse=True)
def _release_cuda_cache():
    """Return a test's cached CUDA blocks after it: on unified memory (the DGX) the caching allocator otherwise keeps
    the largest test's peak for the rest of the session, which the full check's memory claim pays for."""
    yield
    if torch.cuda.is_initialized() and torch.cuda.memory_reserved() > 2**30:
        import gc
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def loaded():
    from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone, load_backbone, resolve_base

    if not Path(resolve_base(None)).exists():
        pytest.skip("LFM2.5-350M is not in the local HF cache")
    torch.manual_seed(0)
    model, tokenizer = load_backbone(dtype=torch.float32, device="cpu")
    backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer))
    return model, tokenizer, backbone
