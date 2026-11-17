#!/usr/bin/env python3
"""Pruned native checkpoints from an external LFM3 HF snapshot, and linear merges of native checkpoints.

    merge_checkpoint.py convert --hf MODEL.safetensors --source fresh-init/checkpoint.pt --out CK
    merge_checkpoint.py merge --base CK0 --term CK1=0.5 [--term CK2=1.0 ...] --out CK

convert: the snapshot's backbone in ``exp0_sft.write_checkpoint``'s pruned layout (BF16, what ``pop_rl_actor.py
serve-checkpoint`` and the held eval serve), with the source's model_config and interface and the source's embedding rows
at the interface ids (the HF model has no learned interface rows). Names, shapes and dtypes must equal the source's.
The safetensors file is read with ``safetensors.safe_open`` only (data, no code).

merge: per tensor, in FP32, ``base + sum_i a_i * (ck_i - base)``, rounded to BF16. One term is the linear interpolation
``base + a*(other - base)``; two terms with a = 1 are task arithmetic (e.g. base + (voltron - base) + (sft - base)). All
inputs must share the base's model_config interface ids and interface_state.

Both record their recipe (inputs with sha256, coefficients, code provenance) as ``merge_recipe`` in the checkpoint and
in ``OUT.json`` beside it.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

INTERFACE_KEYS = ('marker_ids', 'token_override_ids')


def special_ids(state):
    """The embedding rows the interface owns (``exp0_sft.load_model``'s ``special``)."""
    ids = []
    for key in INTERFACE_KEYS:
        ids += [int(i) for i in (state['model_config'].get(key) or ())]
    return ids


def file_sha256(path):
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    return sha256_file_hex_cached(path)


def load_native(path):
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    return CheckpointPolicy.load(Path(path))


def pruned(state, backbone, source_sha256, recipe):
    """``exp0_sft.write_checkpoint``'s output dict for ``backbone`` (tied head re-pointed at the embedding)."""
    if 'lm_head.weight' in state['backbone_state']:
        if not torch.equal(state['backbone_state']['lm_head.weight'], state['backbone_state']['model.embed_tokens.weight']):
            raise ValueError('untied lm_head: tied-embedding checkpoints only')
        backbone['lm_head.weight'] = backbone['model.embed_tokens.weight']
    out = {k: state[k] for k in ('schema', 'model_config', 'interface_state')}
    step = int(state.get('step') or 0)
    out.update(backbone_state=backbone, step=step, inference_only=True, optimizer_pruned=True,
               optimizer_pruned_keys=['training'],
               inference_source={'schema': 'natlang.native-inference-source/1', 'training_checkpoint_sha256': source_sha256,
                                 'training_checkpoint_bytes': 0, 'step': step},
               merge_recipe=recipe)
    return out


def code():
    try:
        from natlang_neuralese.common.code_provenance import code_provenance
        return code_provenance()
    except Exception as error:  # provenance is recorded, never a gate
        return {'error': repr(error)}


def save(out, path, recipe):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(out, tmp)
    tmp.replace(path)
    recipe = dict(recipe, output={'path': str(path), 'sha256': file_sha256(path)})
    Path(str(path) + '.json').write_text(json.dumps(recipe, indent=1) + '\n')
    return recipe


def convert(hf_path, source_path, out_path, *, source_sha256=None, hf_sha256=None):
    from safetensors import safe_open
    state = load_native(source_path)
    source_sha256 = source_sha256 or file_sha256(source_path)
    original = state['backbone_state']
    names = [n for n in original if n != 'lm_head.weight']
    special = torch.tensor(special_ids(state), dtype=torch.long)
    backbone, moved = {}, {}
    with safe_open(str(hf_path), 'pt') as f:
        if 'lm_head.weight' in f.keys() and not torch.equal(f.get_tensor('lm_head.weight'),
                                                             f.get_tensor('model.embed_tokens.weight')):
            raise ValueError('the HF snapshot has an untied lm_head: tied-embedding checkpoints only')
        hf_names = set(f.keys()) - {'lm_head.weight'}
        if hf_names != set(names):
            raise ValueError(f'tensor names differ: only in HF {sorted(hf_names - set(names))[:8]}, '
                             f'only in source {sorted(set(names) - hf_names)[:8]}')
        for name in names:
            value = f.get_tensor(name)
            if value.shape != original[name].shape or value.dtype != original[name].dtype:
                raise ValueError(f'{name}: HF {tuple(value.shape)} {value.dtype} vs source '
                                 f'{tuple(original[name].shape)} {original[name].dtype}')
            if name == 'model.embed_tokens.weight' and len(special):
                value = value.clone()
                value[special] = original[name][special]
            backbone[name] = value.contiguous()
    recipe = {'schema': 'natlang.native-merge-recipe/1', 'op': 'convert',
              'hf': {'path': str(hf_path), 'sha256': hf_sha256 or file_sha256(hf_path)},
              'source': {'path': str(source_path), 'sha256': source_sha256},
              'interface_rows': 'source', 'code': code()}
    return save(pruned(state, backbone, source_sha256, recipe), out_path, recipe)


def merge(base_path, terms, out_path):
    """``terms``: [(path, coefficient)]."""
    base = load_native(base_path)
    others = [(load_native(p), float(a)) for p, a in terms]
    names = [n for n in base['backbone_state'] if n != 'lm_head.weight']
    for (path, _), (state, _) in zip(terms, others):
        if special_ids(state) != special_ids(base):
            raise ValueError(f'{path}: interface ids differ from the base')
        for key, value in base['interface_state'].items():
            other = state['interface_state'].get(key)
            if torch.is_tensor(value) and (other is None or not torch.equal(value, other)):
                raise ValueError(f'{path}: interface_state[{key!r}] differs from the base')
        theirs = set(state['backbone_state']) - {'lm_head.weight'}
        if theirs != set(names):
            raise ValueError(f'{path}: backbone names differ from the base')
    backbone = {}
    for name in names:
        b = base['backbone_state'][name]
        acc = b.float()
        for state, a in others:
            x = state['backbone_state'][name]
            if x.shape != b.shape:
                raise ValueError(f'{name}: shape {tuple(x.shape)} vs base {tuple(b.shape)}')
            acc = acc + a * (x.float() - b.float())
        backbone[name] = acc.to(b.dtype).contiguous()
    base_sha256 = file_sha256(base_path)
    recipe = {'schema': 'natlang.native-merge-recipe/1', 'op': 'merge',
              'formula': 'base + sum_i a_i * (term_i - base), per tensor in FP32, rounded to the base dtype',
              'base': {'path': str(base_path), 'sha256': base_sha256,
                       'merge_recipe': base.get('merge_recipe')},
              'terms': [{'path': str(p), 'sha256': file_sha256(p), 'a': float(a),
                         'merge_recipe': s.get('merge_recipe')} for (p, a), (s, _) in zip(terms, others)],
              'code': code()}
    return save(pruned(base, backbone, base_sha256, recipe), out_path, recipe)


def parse_term(text):
    path, sep, a = text.rpartition('=')
    if not sep or not path:
        raise argparse.ArgumentTypeError('--term CHECKPOINT=COEFFICIENT')
    return path, float(a)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='action', required=True)
    c = commands.add_parser('convert', help='HF safetensors -> pruned native checkpoint')
    c.add_argument('--hf', required=True)
    c.add_argument('--source', required=True, help='native checkpoint of the same architecture (model_config, interface)')
    c.add_argument('--source-sha256', help="the source's known sha256 (skips hashing it)")
    c.add_argument('--out', required=True)
    m = commands.add_parser('merge', help='base + sum a_i (term_i - base)')
    m.add_argument('--base', required=True)
    m.add_argument('--term', type=parse_term, action='append', required=True, metavar='CHECKPOINT=A')
    m.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.action == 'convert':
        recipe = convert(args.hf, args.source, args.out, source_sha256=args.source_sha256)
    else:
        recipe = merge(args.base, args.term, args.out)
    print(json.dumps({k: v for k, v in recipe.items() if k != 'code'}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
