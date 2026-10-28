"""LFM3 deploy exports of the quantization component's points, in the layouts of Liquid's own releases (owner
2026-10-28), so our checkpoints load wherever ``LiquidAI/LFM3-1.7B-NVFP4`` and ``-FP8`` load (vLLM, SGLang, TensorRT):

- ``nvfp4`` (ModelOpt, ``LiquidAI/LFM3-1.7B-NVFP4@b30025bb``): feed_forward w1/w2/w3 as packed E2M1 ``weight`` (uint8),
  E4M3 ``weight_scale`` (group 16), FP32 ``weight_scale_2`` (w1/w3 share it: one fused gate/up GEMM) and the static
  FP32 ``input_scale`` = calibrated amax / (6·448); every other tensor BF16. ``config.json`` carries ModelOpt's
  ``quantization_config`` and ``hf_quant_config.json`` its export record; the ignore list names each layer's
  ``conv*`` or ``self_attn*`` modules, ``lm_head`` and ``model.embed_tokens``.
- ``fp8_block`` (compressed-tensors ``FP8_BLOCK``, ``LiquidAI/LFM3-1.7B-FP8@243955da``): every Linear weight not
  ignored (attention q/k/v/o and feed_forward) as E4M3 ``weight`` with BF16 128x128 ``weight_scale``; activations
  dynamic per group of 128 (no stored scale); ignored: lm_head, embed_tokens, ``.gate``, conv/short_conv modules,
  ``self_attn.gate_proj``.

Selection follows each format's own ignore semantics (ModelOpt: fnmatch on module names; compressed-tensors: exact
names or ``re:`` patterns), applied to the 2-D linear weights of the checkpoint, so the written config is the rule
that chose the tensors. The codecs are train/qformats.py, the same functions the QAT forward uses.

CLI::

    python -m natlang_neuralese.export.lfm3_quantized export --model BF16_DIR --format nvfp4 \\
        --calibration calibration.json --out OUT
    python -m natlang_neuralese.export.lfm3_quantized compare --ours OUT --theirs LIQUID_DIR --format nvfp4
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
from pathlib import Path

import torch

from ..train import qformats as F

MODELOPT_PRODUCER = {'name': 'modelopt', 'version': '0.0.1.dev1+g4c1f81382'}
FP8_BLOCK_CONFIG = {
    'config_groups': {'FP8_BLOCK': {
        'format': 'float-quantized',
        'input_activations': {'actorder': None, 'block_structure': None, 'dynamic': True, 'group_size': 128,
                              'num_bits': 8, 'observer': None, 'observer_kwargs': {}, 'scale_dtype': None,
                              'strategy': 'group', 'symmetric': True, 'type': 'float', 'zp_dtype': None},
        'output_activations': None,
        'targets': ['Linear'],
        'weights': {'actorder': None, 'block_structure': [128, 128], 'dynamic': False, 'group_size': None,
                    'num_bits': 8, 'observer': None, 'observer_kwargs': {}, 'scale_dtype': None, 'strategy': 'block',
                    'symmetric': True, 'type': 'float', 'zp_dtype': None}}},
    'format': 'float-quantized',
    'global_compression_ratio': None,
    'ignore': ['lm_head', 'model.embed_tokens', 're:.*\\.gate$', 're:.*\\.(?:conv|short_conv)\\..*',
               're:.*\\.self_attn\\.gate_proj$'],
    'kv_cache_scheme': None,
    'quant_method': 'compressed-tensors',
    'quantization_status': 'compressed',
    'transform_config': {},
    'version': '0.19.0'}
SHARED_GLOBAL = (('w1', 'w3'),)  # projections fused into one deploy GEMM (gate/up): one NVFP4 global scale
WEIGHT_FILES = ('model.safetensors', 'model.safetensors.index.json', 'config.json')


# Quantization configs --------------------------------------------------------------------------------------------
def nvfp4_exclude(layer_types):
    """ModelOpt's exclude list: lm_head, the embeddings, and each layer's unquantized operator (conv or self_attn),
    sorted as ModelOpt writes it."""
    per_layer = [f'model.layers.{i}.{"conv" if kind == "conv" else "self_attn"}*' for i, kind in enumerate(layer_types)]
    return ['lm_head', 'model.embed_tokens'] + sorted(per_layer)


def nvfp4_configs(layer_types, group=F.NVFP4_GROUP):
    """(config.json ``quantization_config``, ``hf_quant_config.json``) of a ModelOpt NVFP4 export."""
    exclude = nvfp4_exclude(layer_types)
    quantization_config = {
        'config_groups': {'group_0': {
            'input_activations': {'dynamic': False, 'num_bits': 4, 'type': 'float', 'group_size': group},
            'weights': {'dynamic': False, 'num_bits': 4, 'type': 'float', 'group_size': group},
            'targets': ['Linear']}},
        'ignore': exclude, 'quant_algo': 'NVFP4', 'producer': dict(MODELOPT_PRODUCER), 'quant_method': 'modelopt'}
    hf_quant_config = {'producer': dict(MODELOPT_PRODUCER),
                       'quantization': {'quant_algo': 'NVFP4', 'kv_cache_quant_algo': None, 'group_size': group,
                                        'exclude_modules': exclude}}
    return quantization_config, hf_quant_config


def modelopt_ignored(module, patterns):
    return any(fnmatch.fnmatch(module, pattern) for pattern in patterns)


def compressed_tensors_ignored(module, patterns):
    for pattern in patterns:
        if pattern.startswith('re:'):
            if re.match(pattern[3:], module):
                return True
        elif module == pattern:
            return True
    return False


def linear_modules(names_shapes):
    """Module names of the 2-D ``.weight`` tensors (the Linear layers and the embeddings)."""
    return [name[:-len('.weight')] for name, shape in names_shapes if name.endswith('.weight') and len(shape) == 2]


def targets(fmt, names_shapes, layer_types):
    modules = linear_modules(names_shapes)
    if fmt == 'nvfp4':
        exclude = nvfp4_exclude(layer_types)
        return [m for m in modules if not modelopt_ignored(m, exclude)]
    if fmt == 'fp8_block':
        return [m for m in modules if not compressed_tensors_ignored(m, FP8_BLOCK_CONFIG['ignore'])]
    raise ValueError(f'unknown export format: {fmt}')


# Export ----------------------------------------------------------------------------------------------------------
def _module_suffix(name):
    """``…layers.N.feed_forward.w1`` -> ``layers.N.feed_forward.w1`` (calibration names may carry another prefix)."""
    index = name.find('layers.')
    return name[index:] if index >= 0 else name


def _read(model_dir):
    from safetensors import safe_open

    paths = sorted(Path(model_dir).glob('*.safetensors'))
    tensors = {}
    for path in paths:
        with safe_open(str(path), 'pt', 'cpu') as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return tensors


def quantize_state(fmt, tensors, layer_types, input_amax=None, device='cpu', input_scales=None):
    """The exported tensors of ``tensors`` (BF16 HF state) at ``fmt``. ``input_amax`` ({module name: amax}, the
    component's calibration) is required for nvfp4's static input scales; ``input_scales`` ({module name: FP32
    input_scale}) gives them outright instead (another export's, for layout checks)."""
    chosen = targets(fmt, [(k, tuple(v.shape)) for k, v in tensors.items()], layer_types)
    out = {k: v for k, v in tensors.items() if k[:-len('.weight')] not in chosen or not k.endswith('.weight')}
    if fmt == 'fp8_block':
        for module in chosen:
            codes, scales = F.fp8_block_encode(tensors[module + '.weight'].to(device))
            out[module + '.weight'], out[module + '.weight_scale'] = codes.cpu(), scales.cpu()
        return out
    amax_by_suffix = {_module_suffix(k): v for k, v in (input_amax or {}).items()}
    for module in chosen:
        parent, leaf = module.rsplit('.', 1)
        weight = tensors[module + '.weight'].to(device)
        shared = next((s for s in SHARED_GLOBAL if leaf in s), (leaf,))
        global_amax = torch.stack([tensors[f'{parent}.{n}.weight'].to(device).abs().amax() for n in shared
                                   if f'{parent}.{n}.weight' in tensors]).amax()
        packed, scales, scale_2 = F.nvfp4_weight_encode(weight, global_amax=global_amax)
        out[module + '.weight'], out[module + '.weight_scale'] = packed.cpu(), scales.cpu()
        out[module + '.weight_scale_2'] = scale_2.reshape(()).cpu()
        if input_scales is not None:
            out[module + '.input_scale'] = input_scales[module].float().reshape(()).cpu()
            continue
        amax = amax_by_suffix.get(_module_suffix(module))
        if amax is None:
            raise ValueError(f'no calibrated input amax for {module} (nvfp4 input_scale is static)')
        out[module + '.input_scale'] = F.nvfp4_global_scale(torch.tensor(float(amax), dtype=torch.float32)).reshape(())
    return out


def export(model_dir, out_dir, fmt, *, input_amax=None, tensors=None, device='cpu', input_scales=None):
    """Write ``model_dir`` (an HF BF16 LFM3 checkpoint; ``tensors`` overrides its weights) as a ``fmt`` checkpoint in
    ``out_dir``: weights, config with the format's ``quantization_config``, and the model's other files."""
    from safetensors.torch import save_file

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    config = json.loads((model_dir / 'config.json').read_text())
    tensors = _read(model_dir) if tensors is None else tensors
    state = quantize_state(fmt, tensors, config['layer_types'], input_amax, device, input_scales)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in model_dir.iterdir():
        if path.is_file() and path.name not in WEIGHT_FILES and not path.name.endswith('.safetensors'):
            shutil.copyfile(path, out_dir / path.name)
    state = {k: v.contiguous() for k, v in sorted(state.items())}
    if fmt == 'nvfp4':
        quantization_config, hf_quant_config = nvfp4_configs(config['layer_types'])
        save_file(state, str(out_dir / 'model.safetensors'), metadata={'format': 'pt'})
        (out_dir / 'hf_quant_config.json').write_text(json.dumps(hf_quant_config, indent=2) + '\n')
    else:
        quantization_config = json.loads(json.dumps(FP8_BLOCK_CONFIG))
        save_file(state, str(out_dir / 'model.safetensors'))
        total = sum(v.numel() * v.element_size() for v in state.values())
        index = {'metadata': {'total_size': total}, 'weight_map': {k: 'model.safetensors' for k in state}}
        (out_dir / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    (out_dir / 'config.json').write_text(json.dumps({**config, 'quantization_config': quantization_config},
                                                    indent=2) + '\n')
    return {'format': fmt, 'out': str(out_dir), 'tensors': len(state),
            'quantized_modules': len(targets(fmt, [(k, tuple(v.shape)) for k, v in tensors.items()],
                                             config['layer_types']))}


# Comparison ------------------------------------------------------------------------------------------------------
def _dequantize(fmt, tensors, module):
    if fmt == 'nvfp4':
        return F.nvfp4_weight_decode(tensors[module + '.weight'], tensors[module + '.weight_scale'],
                                     tensors[module + '.weight_scale_2'])
    return F.fp8_block_decode(tensors[module + '.weight'], tensors[module + '.weight_scale'])


def _equal_fraction(a, b):
    """Fraction of bit-identical elements (packed FP4 bytes compared per 4-bit code)."""
    if a.dtype == torch.uint8:
        a, b = F.unpack_fp4(a), F.unpack_fp4(b)
    else:
        bits = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        a, b = a.contiguous().reshape(-1).view(bits), b.contiguous().reshape(-1).view(bits)
    return float((a == b).float().mean())


def compare_states(fmt, ours, theirs, layer_types, reference=None):
    """Per-tensor comparison of two exported states: key sets, dtype/shape, exact-equal fraction (codes compared per
    4-bit element), and per quantized module the max |dequantized difference| (and, with ``reference`` BF16 weights,
    each side's relative error to them)."""
    report = {'missing': sorted(set(theirs) - set(ours)), 'extra': sorted(set(ours) - set(theirs)), 'tensors': {},
              'modules': {}}
    for key in sorted(set(ours) & set(theirs)):
        a, b = ours[key], theirs[key]
        if a.dtype != b.dtype or a.shape != b.shape:
            report['tensors'][key] = {'dtype': [str(a.dtype), str(b.dtype)], 'shape': [list(a.shape), list(b.shape)]}
            continue
        report['tensors'][key] = {'equal': _equal_fraction(a, b)}
    for module in sorted(k[:-len('.weight_scale')] for k in theirs if k.endswith('.weight_scale')):
        if module + '.weight_scale' not in ours:
            continue
        a, b = _dequantize(fmt, ours, module), _dequantize(fmt, theirs, module)
        row = {'max_abs_diff': float((a - b).abs().max())}
        if reference is not None:
            w = reference[module + '.weight'].float()
            row['rel_err_ours'] = float((a - w).norm() / w.norm())
            row['rel_err_theirs'] = float((b - w).norm() / w.norm())
        report['modules'][module] = row
    unequal = {k: v for k, v in report['tensors'].items() if v.get('equal') != 1.0}
    report['summary'] = {'tensors': len(report['tensors']), 'tensors_bit_identical': len(report['tensors']) - len(unequal),
                         'unequal': unequal, 'modules': len(report['modules']),
                         'max_dequantized_diff': max((m['max_abs_diff'] for m in report['modules'].values()),
                                                     default=0.0)}
    return report


def compare(ours_dir, theirs_dir, fmt, reference_dir=None):
    ours_dir, theirs_dir = Path(ours_dir), Path(theirs_dir)
    config_ours = json.loads((ours_dir / 'config.json').read_text())
    config_theirs = json.loads((theirs_dir / 'config.json').read_text())
    reference = _read(reference_dir) if reference_dir else None
    report = compare_states(fmt, _read(ours_dir), _read(theirs_dir), config_theirs['layer_types'], reference)
    report['quantization_config_equal'] = config_ours.get('quantization_config') == config_theirs.get('quantization_config')
    report['config_other_keys'] = sorted(k for k in set(config_ours) | set(config_theirs)
                                         if k != 'quantization_config' and config_ours.get(k) != config_theirs.get(k))
    if fmt == 'nvfp4':
        report['hf_quant_config_equal'] = (json.loads((ours_dir / 'hf_quant_config.json').read_text())
                                           == json.loads((theirs_dir / 'hf_quant_config.json').read_text()))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    out = sub.add_parser('export', help='write an HF BF16 LFM3 checkpoint in a deploy format')
    out.add_argument('--model', type=Path, required=True)
    out.add_argument('--format', choices=('nvfp4', 'fp8_block'), required=True)
    out.add_argument('--calibration', type=Path, help='Quantization.state_dict() JSON (nvfp4 input amax)')
    out.add_argument('--input-scales-from', type=Path,
                     help='take nvfp4 input scales from another NVFP4 export (layout checks against a reference)')
    out.add_argument('--out', type=Path, required=True)
    out.add_argument('--device', default='cpu')
    cmp = sub.add_parser('compare', help='compare two exports tensor by tensor')
    cmp.add_argument('--ours', type=Path, required=True)
    cmp.add_argument('--theirs', type=Path, required=True)
    cmp.add_argument('--format', choices=('nvfp4', 'fp8_block'), required=True)
    cmp.add_argument('--reference', type=Path, help='the BF16 model the exports quantize (relative errors)')
    cmp.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'export':
        amax = scales = None
        if args.calibration:
            amax = json.loads(args.calibration.read_text())['amax']
        elif args.input_scales_from:
            scales = {k[:-len('.input_scale')]: v for k, v in _read(args.input_scales_from).items()
                      if k.endswith('.input_scale')}
        print(json.dumps(export(args.model, args.out, args.format, input_amax=amax, device=args.device,
                                input_scales=scales)))
        return 0
    report = compare(args.ours, args.theirs, args.format, args.reference)
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in report if k not in ('tensors', 'modules')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
