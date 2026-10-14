#!/usr/bin/env python3
"""Read-only mask, rendering, and teacher-forced action audit for one admitted Luna case."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.eval.task_cadence import restore_engine
from natlang_neuralese.train.direct_sft import context_only_mask, text_engine
from natlang_neuralese.train.recipe import resolve_recipe_data
from natlang_neuralese.train.text_supervision import ROLE_CODES, balanced_position_weights
from natlang_neuralese.train.text_warmup import prepare_text_windows


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def selected(path, ids):
    found = {}
    for line in path.read_bytes().splitlines():
        row = json.loads(line)
        if row.get('id') in ids:
            found[row['id']] = (row, hashlib.sha256(line).hexdigest())
    if set(found) != set(ids):
        raise ValueError('selected IDs absent from native source')
    return found


def data_audit(engine, row, params):
    windows, receipt = prepare_text_windows(
        engine, [row], tokens=params['tokens'], prefix_tokens=params['prefix_tokens'],
        target_tokens=params['target_tokens'], train_target_tokens=params['train_target_tokens'],
        train_min_context_tokens=params['train_min_context_tokens'], mask_system_prompt=True)
    chunks = []
    for window in windows['train']:
        ids = list(window['ids'])
        roles = list(window['roles'])
        source = window['ids']._source
        labels = window['roles']._source
        failed = context_only_mask(source, labels, row.get('context_only_spans', ()), engine.tokenizer.decode)
        excluded = torch.tensor([failed[i] for i in range(window['start'], window['start'] + len(ids))], dtype=torch.bool)
        prefix = window['prefix']
        span = torch.tensor([ids[prefix:]], dtype=torch.long)
        role_span = torch.tensor([roles[prefix:]], dtype=torch.long)
        weights = balanced_position_weights(
            span, [window.get('supervised_suffix_start')], roles=role_span,
            context_weight=params['context_weight'], feedback_weight=params['feedback_weight'],
            exclude=excluded[prefix:][None])[0]
        suffix_start = window.get('supervised_suffix_start')
        target_start = prefix + suffix_start if suffix_start is not None else len(ids)
        chunks.append({
            'window_start': window['start'], 'window_tokens': len(ids), 'prefix_tokens': prefix,
            'target_start': target_start, 'target_tokens': max(0, len(ids) - target_start - 1),
            'target_excluded': int(excluded[target_start:-1].sum()),
            'target_weight_sum': float(weights[suffix_start:-1].sum()) if suffix_start is not None else 0,
            'total_weight_sum': float(weights.sum()),
            'role_weight_sum': {name: float(weights[role_span[0] == code].sum())
                                for code, name in enumerate(ROLE_CODES)},
            'role_positions': {name: int((role_span[0] == code).sum())
                               for code, name in enumerate(ROLE_CODES)},
            'excluded_assistant_positions': int(sum(bool(excluded[i]) and roles[i] in (4, 5)
                                                   for i in range(prefix, len(ids)))),
        })
    return {'receipt': receipt, 'windows': chunks}


@torch.inference_mode()
def score_gold(engine, row):
    ids = row['token_ids']
    start = row['supervised_suffix_start']
    device = engine.backbone.embedding_weight.device
    input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    out = engine.backbone.forward_ids(input_ids, cutoff=engine.heads.cutoff, logits=False)
    hidden = out['h_final'][:, start - 1:]
    logits, _ = engine.advance_token_readout(None, hidden, last_only=False)
    gold = torch.tensor(ids[start:], dtype=torch.long, device=device)
    if logits.shape[1] != gold.numel():
        raise ValueError(f'logit/gold length mismatch: {logits.shape[1]} != {gold.numel()}')
    log_probs = logits.float().log_softmax(-1)[0]
    nll = -log_probs.gather(1, gold[:, None]).squeeze(1)
    agree = logits[0].argmax(-1) == gold
    disagreement = (~agree).nonzero().flatten()
    return {'positions': int(gold.numel()), 'mean_nll': float(nll.mean()),
            'argmax_agreement': float(agree.float().mean()),
            'first_argmax_disagreement': None if not len(disagreement) else int(disagreement[0]),
            'first_16_nll': [float(x) for x in nll[:16]],
            'last_16_nll': [float(x) for x in nll[-16:]],
            'gold_suffix_sha256': digest(ids[start:])}


def serving_boundary(engine, native, row):
    engine.prompt_rendering = row['prompt_rendering']
    plan = engine._prompt_plan(native['messages'], native['tools'])
    if any(kind != 'tokens' for kind, _ in plan):
        return {'status': 'neuralese-blocks-in-live-prompt'}
    live = [token for _, segment in plan for token in segment]
    train = row['token_ids'][:row['supervised_suffix_start']]
    first = next((i for i, (a, b) in enumerate(zip(live, train)) if a != b), min(len(live), len(train)))
    return {'status': 'exact' if live == train else 'different',
            'live_tokens': len(live), 'train_prefix_tokens': len(train),
            'first_difference': None if live == train else first,
            'live_token_ids_sha256': digest(live), 'train_prefix_token_ids_sha256': digest(train),
            'live_near_difference': engine.tokenizer.decode(live[max(0, first-12):first+12], skip_special_tokens=False),
            'train_near_difference': engine.tokenizer.decode(train[max(0, first-12):first+12], skip_special_tokens=False)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--render', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    native_path = Path(selection['source'])
    if sha256_file_hex(native_path) != selection['source_sha256']:
        raise ValueError('native source SHA mismatch')
    natives = selected(native_path, selection['ids'])
    if [natives[row_id][1] for row_id in selection['ids']] != selection['source_row_sha256']:
        raise ValueError('native row SHA mismatch')
    recipe = resolve_recipe_data(args.recipe)
    defaults = recipe['stage_parameter_defaults']['neuralese_direct']
    stage = next(stage for stage in recipe['stages'] if stage['id'] == 'direct_sft')
    params = {**defaults, **stage.get('parameters', {})}
    reader = text_engine(args.heads)
    if args.gpu:
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'request.json').write_text(json.dumps({'weights': str(args.checkpoint.resolve())}))
            engine = restore_engine({'dir': temp, 'heads': str(args.heads.resolve()), 'device': 'cuda',
                                     'prompt_rendering': 'full'})
            report = run(args, selection, natives, params, engine, True)
    else:
        report = run(args, selection, natives, params, reader, False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'out': str(args.out), 'cases': len(report['cases']), 'gpu': args.gpu}))


def run(args, selection, natives, params, engine, gpu):
    report = {'schema': 'natlang.qad-action-target-audit/1',
              'selection_sha256': sha256_file_hex(args.selection),
              'recipe_sha256': sha256_file_hex(args.recipe),
              'checkpoint_sha256': sha256_file_hex(args.checkpoint),
              'heads_sha256': sha256_file_hex(args.heads),
              'parameters': {key: params[key] for key in ('tokens', 'prefix_tokens', 'target_tokens',
                  'train_target_tokens', 'train_min_context_tokens', 'context_weight', 'feedback_weight')},
              'cases': []}
    for mode in ('program-header', 'full'):
        path = args.render / mode / 'lfm25' / 'text-data.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for row in rows:
            native, native_sha = natives[row['id']]
            case = {'id': row['id'], 'mode': mode, 'native_row_sha256': native_sha,
                    'render_sha256': sha256_file_hex(path),
                    'data': data_audit(engine, row, params)}
            if gpu:
                case['serving_boundary'] = serving_boundary(engine, native, row)
                case['gold_scoring'] = score_gold(engine, row)
            report['cases'].append(case)
    return report


if __name__ == '__main__':
    main()
