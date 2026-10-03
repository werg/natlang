"""Parity of the llama.cpp Neuralese fork with the PyTorch reference (S4 §6).

Runs the fork's `llama-neuralese-parity` on a model GGUF and a projector GGUF exported from the
reference, on CPU in float32, and compares: read-port logits and greedy continuation; the write
procedure's shallow residuals, completed residuals, payload (temperature 0), stop logits,
read-back logits and greedy continuation.

Skipped unless the fork is built (see training/neuralese/llama-cpp-fork.json).
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

STEPS = 8


def _binary():
    from natlang_neuralese.export import fork_root

    path = fork_root() / "build-cpu" / "bin" / "llama-neuralese-parity"
    if not path.exists():
        pytest.skip(f"llama.cpp Neuralese fork not built at {path}")
    return path


def _write_case(path: Path, arrays: dict):
    with open(path, "wb") as fp:
        for name, (dtype, values) in arrays.items():
            data = list(values)
            fp.write(struct.pack("<i", len(name)) + name.encode() + struct.pack("<ii", dtype, len(data)))
            fp.write(struct.pack(f"<{len(data)}{'i' if dtype == 0 else 'f'}", *data))


def _read_result(path: Path) -> dict:
    out, raw = {}, Path(path).read_bytes()
    i = 0
    while i < len(raw):
        (n,) = struct.unpack_from("<i", raw, i); i += 4
        name = raw[i:i + n].decode(); i += n
        dtype, count = struct.unpack_from("<ii", raw, i); i += 8
        values = struct.unpack_from(f"<{count}{'i' if dtype == 0 else 'f'}", raw, i); i += 4 * count
        out[name] = torch.tensor(values, dtype=torch.int64 if dtype == 0 else torch.float32)
    return out


@pytest.fixture(scope="module")
def port(loaded, tmp_path_factory):
    """Perturbed heads (so every path carries signal), exported to GGUF."""
    from natlang_neuralese.export import export_heads_gguf, export_model_gguf, export_model_hf
    from natlang_neuralese.model.heads import PortHeads

    binary = _binary()
    model, tokenizer, backbone = loaded
    torch.manual_seed(7)
    heads = PortHeads(backbone, cutoff=6, max_length=6).eval()
    with torch.no_grad():
        for name, p in heads.named_parameters():
            if name.startswith("feedback.readout.") or name == "feedback.gate":
                continue
            p.add_(0.02 * torch.randn_like(p))
        heads.stop.mlp_out.bias.fill_(-0.2)  # some blocks stop before the hard maximum
    out = tmp_path_factory.mktemp("nz-gguf")
    hf_dir = export_model_hf(backbone, tokenizer, out / "hf")
    model_gguf = export_model_gguf(hf_dir, out / "model-f32.gguf")
    heads_gguf = export_heads_gguf(heads, backbone, out / "neuralese-f32.gguf")
    return {"binary": binary, "model": model_gguf, "heads": heads_gguf, "backbone": backbone, "heads_module": heads,
            "tokenizer": tokenizer, "dir": out}


def _run(port, case: dict, name: str) -> dict:
    case_path, out_path = port["dir"] / f"{name}.case", port["dir"] / f"{name}.out"
    _write_case(case_path, case)
    subprocess.run([str(port["binary"]), "-m", str(port["model"]), "--nz", str(port["heads"]),
                    "--case", str(case_path), "--out", str(out_path), "-t", "8"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return _read_result(out_path)


def _prefix(port, text: str) -> list[int]:
    ids = port["tokenizer"](text, add_special_tokens=False)["input_ids"]
    return [port["tokenizer"].bos_token_id or 1] + ids + [port["backbone"].controls.open_id]


def _close(a, b, atol, what):
    diff = (a.float() - b.float()).abs().max().item()
    assert diff <= atol, f"{what}: max |diff| {diff:.3g} > {atol}"


def test_read_port_matches_reference(port):
    from natlang_neuralese.write import greedy_continue

    backbone, heads, tok = port["backbone"], port["heads_module"], port["tokenizer"]
    prefix = _prefix(port, "Remember this note:")
    suffix = [backbone.controls.close_id] + tok(" The note says", add_special_tokens=False)["input_ids"]
    torch.manual_seed(3)
    source = tok(" the meeting moved to Thursday at noon", add_special_tokens=False)["input_ids"]
    payload = backbone.embed(torch.tensor([source])).detach()[0] + 0.01 * torch.randn(len(source), backbone.embedding_weight.shape[1])

    with torch.no_grad():
        embeds = torch.cat([backbone.embed(torch.tensor([prefix])), heads.interface(payload)[None],
                            backbone.embed(torch.tensor([suffix]))], dim=1)
        out = backbone.forward_embeds(embeds)
        tokens, _ = greedy_continue(backbone, out["cache"], out["logits"][:, -1], STEPS)
    ref_logits = out["logits"][0, -1]

    got = _run(port, {"mode": (0, [0]), "prefix": (0, prefix), "payload": (1, payload.flatten().tolist()),
                      "suffix": (0, suffix), "steps": (0, [STEPS])}, "read")
    _close(got["logits"], ref_logits, 2e-2, "read-port logits")
    assert got["logits"].argmax().item() == ref_logits.argmax().item()
    assert got["greedy"].tolist() == tokens


def test_write_procedure_matches_reference(port):
    from natlang_neuralese.write import greedy_continue, open_block, read_back, write_block

    backbone, heads = port["backbone"], port["heads_module"]
    prefix = _prefix(port, "Summarise the plan in a block:")
    with torch.no_grad():
        opened = open_block(backbone, heads, torch.tensor([prefix]))
        written = write_block(backbone, heads, opened, max_length=heads.max_length)
        back = read_back(backbone, heads, written.block_start, written.payload)
        tokens, _ = greedy_continue(backbone, back["cache"], back["logits"], STEPS)
    n = int(written.lengths[0])

    got = _run(port, {"mode": (0, [1]), "prefix": (0, prefix), "max_length": (0, [heads.max_length]),
                      "steps": (0, [STEPS])}, "write")
    d = backbone.embedding_weight.shape[1]
    assert int(got["n"][0]) == n
    assert bool(got["truncated"][0]) == bool(written.truncated[0])
    _close(got["h_cut"], opened.h_cut[0], 2e-3, "h_cut")
    _close(got["sketches"].view(n, d), written.sketches[0, :n], 2e-3, "sketches")
    _close(got["shallow"].view(n, d), written.shallow[0, :n], 5e-3, "shallow residuals")
    _close(got["final"].view(n, d), written.final[0, :n], 2e-2, "completed residuals")
    _close(got["payload"].view(n, d), written.payload[0, :n], 2e-2, "payload")
    _close(got["log_sigma"].view(n, d), written.log_sigma[0, :n], 1e-2, "log sigma")
    _close(got["stop_logits"], written.stop_logits[0, :got["stop_logits"].numel()], 2e-3, "stop logits")
    _close(got["logits"], back["logits"][0], 5e-2, "read-back logits")
    assert got["greedy"].tolist() == tokens
