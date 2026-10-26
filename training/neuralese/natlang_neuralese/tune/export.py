"""PEFT safetensors and GGUF export of one static effective stack (§5.3, §7).

Convention: a single global ``lora_alpha`` with per-module ranks. Servers apply
``alpha / r_m`` per module (PEFT via ``rank_pattern``; llama.cpp per tensor rank),
so ``r_m / alpha`` is folded into ``lora_B`` exactly once and the effective
delta is ``lora_B @ lora_A``. Native names map to HF modules only through an
explicit table; an unmapped nonzero target is an error, never an omission.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex
from . import TuneMathError

EXPORTER_VERSION = 'natlang.tune.peft-export/1'
LLAMA_CPP = Path(__file__).resolve().parents[4] / 'vendor' / 'llama.cpp'


@dataclass(frozen=True)
class ModuleTarget:
    hf_module: str           # may contain '{expert}' for a batched expert bank
    orientation: str = 'out_in'  # native weight layout: 'out_in' (nn.Linear) or 'in_out'


def _modules(name, factors, target):
    """Yield ``(hf_module, lora_A (r, in), lora_B (out, r))`` per expert."""
    B, A = factors.B.detach().float().cpu(), factors.A.detach().float().cpu()
    if target.orientation == 'in_out':
        B, A = A.transpose(-2, -1), B.transpose(-2, -1)
    elif target.orientation != 'out_in':
        raise TuneMathError(f'unknown orientation for {name}: {target.orientation}')
    if B.ndim == 2:
        if '{expert}' in target.hf_module:
            raise TuneMathError(f'{name} is a single matrix but its module template is per expert')
        yield target.hf_module, A, B
    elif B.ndim == 3:
        if '{expert}' not in target.hf_module:
            raise TuneMathError(f'{name} is an expert bank; its module template needs an {{expert}} field')
        for expert in range(B.shape[0]):
            yield target.hf_module.format(expert=expert), A[expert], B[expert]
    else:
        raise TuneMathError(f'unsupported target rank layout for {name}')


def export_peft(stack, module_map, out_dir, *, base_identity, base_model_name_or_path, lora_alpha=1.0, stack_identity=None):
    """Write ``adapter_model.safetensors``, ``adapter_config.json`` and ``natlang-export-receipt.json``."""
    from safetensors.torch import save_file
    unmapped = sorted(name for name, factors in stack.items() if factors.rank and name not in module_map)
    if unmapped:
        raise TuneMathError(f'no HF module mapping for nonzero targets: {unmapped}')
    tensors, ranks, modules, zero = {}, {}, {}, []
    for name in sorted(stack):
        factors = stack[name]
        if not factors.rank:
            zero.append(name)
            continue
        for module, A, B in _modules(name, factors, module_map[name]):
            if module in ranks:
                raise TuneMathError(f'two targets map to HF module {module}')
            rank = A.shape[0]
            tensors[f'base_model.model.{module}.lora_A.weight'] = A.contiguous()
            tensors[f'base_model.model.{module}.lora_B.weight'] = (B * (rank / lora_alpha)).contiguous()
            ranks[module] = rank
            modules.setdefault(name, []).append(module)
    if not tensors:
        raise TuneMathError('stack has no nonzero target to export')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / 'adapter_model.safetensors')
    config = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'base_model_name_or_path': base_model_name_or_path,
              'r': max(ranks.values()), 'lora_alpha': lora_alpha, 'rank_pattern': ranks, 'alpha_pattern': {},
              'target_modules': sorted(ranks), 'lora_dropout': 0.0, 'bias': 'none', 'fan_in_fan_out': False,
              'use_rslora': False, 'use_dora': False, 'inference_mode': True}
    (out / 'adapter_config.json').write_text(json.dumps(config, indent=1, sort_keys=True))
    receipt = {'schema': 'natlang.tune.export-receipt/1', 'exporter': EXPORTER_VERSION, 'base_identity': base_identity,
               'stack_identity': stack_identity, 'scale_convention': 'global lora_alpha; r_m/alpha folded into lora_B; effective delta = lora_B @ lora_A',
               'lora_alpha': lora_alpha, 'module_map': modules, 'ranks': ranks,
               'modes': {name: stack[name].mode for name in modules}, 'zero_targets': zero,
               'contributors': {name: list(stack[name].contributors) for name in modules},
               'safetensors_sha256': sha256_file_hex(out / 'adapter_model.safetensors'),
               'config_sha256': sha256_file_hex(out / 'adapter_config.json')}
    receipt['receipt_sha256'] = canonical_json_sha256_hex(receipt)
    (out / 'natlang-export-receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True))
    return receipt


def load_peft_dense(out_dir):
    """Reconstruct ``{hf_module: delta (out, in)}`` exactly as a PEFT/llama.cpp server applies it."""
    from safetensors.torch import load_file
    out = Path(out_dir)
    config = json.loads((out / 'adapter_config.json').read_text())
    tensors = load_file(out / 'adapter_model.safetensors')
    deltas = {}
    for module in config['target_modules']:
        A = tensors[f'base_model.model.{module}.lora_A.weight']
        B = tensors[f'base_model.model.{module}.lora_B.weight']
        rank = config['rank_pattern'].get(module, config['r'])
        if A.shape[0] != rank:
            raise TuneMathError(f'exported rank for {module} differs from its config')
        deltas[module] = (config['lora_alpha'] / rank) * (B.double() @ A.double())
    return deltas


def export_gguf(peft_dir, base_config_dir, outfile, *, outtype='f32'):
    """Convert a PEFT export with the vendored llama.cpp ``convert_lora_to_gguf.py``.

    ``base_config_dir`` holds the base model's HF config/tokenizer. Architecture
    coverage of this repository's native targets (all LFM projections, Mellum
    expert layouts) is backend work verified by parity tests, not implied here.
    """
    script = LLAMA_CPP / 'convert_lora_to_gguf.py'
    if not script.is_file():
        raise NotImplementedError(f'GGUF LoRA export needs {script}, which is absent from vendor/llama.cpp')
    command = [sys.executable, str(script), '--outtype', outtype, '--outfile', str(outfile),
               '--base', str(base_config_dir), str(peft_dir)]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(LLAMA_CPP), str(LLAMA_CPP / 'gguf-py'), os.environ.get('PYTHONPATH', '')])}
    result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=LLAMA_CPP)
    if result.returncode != 0:
        raise TuneMathError(f'GGUF LoRA conversion failed: {result.stderr[-2000:]}')
    return {'schema': 'natlang.tune.gguf-receipt/1', 'converter': str(script.relative_to(LLAMA_CPP.parents[1])),
            'outtype': outtype, 'peft_receipt_sha256': json.loads((Path(peft_dir) / 'natlang-export-receipt.json').read_text())['receipt_sha256'],
            'gguf_sha256': sha256_file_hex(outfile)}
