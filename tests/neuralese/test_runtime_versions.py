"""The browser runtime (vendored wasm) is the fork's neuralese-service.cpp compiled to WebAssembly, so its API is
the native server's by construction; only its build can lag. Pin both to one fork commit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_vendored_wasm_is_built_from_the_pinned_fork_commit():
    pin = json.loads((ROOT / "training/neuralese/llama-cpp-fork.json").read_text())["commit"]
    provenance = json.loads((ROOT / "ts-host/vendor/neuralese-wasm/provenance.json").read_text())
    built = provenance["fork_commit"]
    assert pin.startswith(built) or built.startswith(pin), (
        f"browser runtime built from fork {built}, native pin is {pin}: rebuild with tools/neuralese/wasm/build.sh "
        "(plain, THREADS=1, GPU=1) and update ts-host/vendor/neuralese-wasm")
