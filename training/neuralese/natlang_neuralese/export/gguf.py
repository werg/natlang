"""Export a trained port to the llama.cpp Neuralese fork (S4 §6).

Two files, like llama.cpp's multimodal projectors:

- the **model GGUF**: the backbone with its trained control-token rows merged into the (tied)
  embedding, converted by the fork's unchanged `convert_hf_to_gguf.py`;
- the **projector GGUF** (`general.architecture = "neuralese"`): interface norm, feedback
  projection, stop head and content projection, plus cutoff, maximum length, control-token IDs
  and dialect. The fork's `tools/neuralese` library loads it (`nz_heads_load`).

The fork's location comes from `NATLANG_LLAMA_NEURALESE`, else the pin file
`training/neuralese/llama-cpp-fork.json`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ..model.dialect import DIALECT
from ..model.heads import PortHeads
from ..model.lfm2_port import PortBackbone

PIN_FILE = Path(__file__).resolve().parents[2] / "llama-cpp-fork.json"


def fork_root() -> Path:
    env = os.environ.get("NATLANG_LLAMA_NEURALESE")
    if env:
        return Path(env)
    path = Path(json.loads(PIN_FILE.read_text())["path"])
    return path if path.is_absolute() else (PIN_FILE.parents[2] / path).resolve()


def _gguf_module():
    gguf_py = fork_root() / "gguf-py"
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))
    import gguf  # noqa: PLC0415

    return gguf


def export_model_hf(backbone: PortBackbone, tokenizer, out_dir: str | Path) -> Path:
    """Save the backbone as an HF checkpoint with the control rows merged into the embedding."""
    out_dir = Path(out_dir)
    weight = backbone.embedding_weight
    ids = [backbone.controls.open_id, backbone.controls.close_id]
    saved = weight.data[ids].clone()
    with torch.no_grad():
        weight.data[ids] = backbone.control_rows.data.to(weight.dtype)
    try:
        backbone.hf.save_pretrained(out_dir)
    finally:
        with torch.no_grad():
            weight.data[ids] = saved
    tokenizer.save_pretrained(out_dir)
    return out_dir


def export_model_gguf(hf_dir: str | Path, out_file: str | Path, outtype: str = "f32") -> Path:
    """Convert an HF checkpoint with the fork's converter."""
    root = fork_root()
    env = dict(os.environ, PYTHONPATH=str(root / "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run(
        [sys.executable, str(root / "convert_hf_to_gguf.py"), str(hf_dir), "--outfile", str(out_file),
         "--outtype", outtype],
        check=True, env=env, stdout=subprocess.DEVNULL,
    )
    return Path(out_file)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().contiguous().numpy()


def export_heads_gguf(heads: PortHeads, backbone: PortBackbone, out_file: str | Path, dialect: str = DIALECT) -> Path:
    """Write the projector GGUF read by the fork's `nz_heads_load`."""
    gguf = _gguf_module()
    eps = float(backbone.config.norm_eps)
    writer = gguf.GGUFWriter(str(out_file), "neuralese")
    writer.add_uint32("neuralese.n_embd", int(backbone.embedding_weight.shape[1]))
    writer.add_uint32("neuralese.cutoff", int(heads.cutoff))
    writer.add_uint32("neuralese.max_length", int(heads.max_length))
    writer.add_uint32("neuralese.open_token_id", int(backbone.controls.open_id))
    writer.add_uint32("neuralese.close_token_id", int(backbone.controls.close_id))
    writer.add_float32("neuralese.norm_eps", eps)
    writer.add_float32("neuralese.feedback.tau", float(heads.feedback.tau))
    writer.add_string("neuralese.dialect", dialect)
    for module, eps_module in [(heads.interface, heads.interface.eps), (heads.feedback.readout_norm, heads.feedback.readout_norm.eps),
                               (heads.feedback.mlp_norm, heads.feedback.mlp_norm.eps), (heads.stop.norm, heads.stop.norm.eps),
                               (heads.content.norm, heads.content.norm.eps)]:
        if abs(eps_module - eps) > 1e-12:
            raise ValueError("all port norms must share the backbone's norm_eps")

    fb, stop, ct = heads.feedback, heads.stop, heads.content
    tensors = {
        "nz.interface.weight": heads.interface.weight,
        "nz.feedback.readout_norm.weight": fb.readout_norm.weight,
        "nz.feedback.readout.weight": fb.readout.weight,          # [V, d]
        "nz.feedback.table_t": fb.embedding.t(),                  # [d, V]: ggml ne = (V, d)
        "nz.feedback.mlp_norm.weight": fb.mlp_norm.weight,
        "nz.feedback.mlp_in.weight": fb.mlp_in.weight,
        "nz.feedback.mlp_in.bias": fb.mlp_in.bias,
        "nz.feedback.mlp_out.weight": fb.mlp_out.weight,
        "nz.feedback.mlp_out.bias": fb.mlp_out.bias,
        "nz.feedback.gate": fb.gate.reshape(1),
        "nz.stop.norm.weight": stop.norm.weight,
        "nz.stop.position.weight": stop.position.weight,          # [max_length + 1, pos_dim]
        "nz.stop.mlp_in.weight": stop.mlp_in.weight,
        "nz.stop.mlp_in.bias": stop.mlp_in.bias,
        "nz.stop.mlp_out.weight": stop.mlp_out.weight,
        "nz.stop.mlp_out.bias": stop.mlp_out.bias,
        "nz.content.norm.weight": ct.norm.weight,
        "nz.content.proj.weight": ct.proj.weight,
        "nz.content.proj.bias": ct.proj.bias,
        "nz.content.log_sigma.weight": ct.log_sigma.weight,
        "nz.content.log_sigma.bias": ct.log_sigma.bias,
    }
    for name, tensor in tensors.items():
        writer.add_tensor(name, _np(tensor))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return Path(out_file)
