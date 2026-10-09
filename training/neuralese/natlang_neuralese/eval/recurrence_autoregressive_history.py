"""Run the shared held autoregressive-history controls on one exact recurrence checkpoint.

The rollout math is delegated to ``projected_history.autoregressive_history_metrics``.
This runner only rebuilds and pins the same source-disjoint held text windows used by
the named warm-up plan, then records one independently scored row per selected window.
It is a diagnostic, not a runtime, foundation, task, or stopping certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..serve.recurrence_checkpoint import load_recurrence_checkpoint
from ..train.output_embedding_projection import sha
from ..train.text_warmup import (
    ROLE_CODES,
    chat_roles,
    document_windows,
    evaluation_batches,
    load_text_rows,
    select_held_document_windows,
)
from .projected_history import autoregressive_history_metrics


def _special_id(tokenizer, token):
    try:
        value = tokenizer.convert_tokens_to_ids(token)
        return value if isinstance(value, int) and value != tokenizer.unk_token_id else None
    except Exception:
        return None


def _held_windows(engine, records, pieces, text_data, *, tokens, prefix_tokens,
                  held_documents, mask_system_prompt):
    rows, receipt = load_text_rows(records, pieces, text_data, tokenizer=engine.tokenizer)
    tokenizer = engine.tokenizer
    role_start = _special_id(tokenizer, '<|im_start|>')
    role_ids = ({i: name for name in ('system', 'user', 'assistant', 'tool')
                 for i in [_special_id(tokenizer, name)] if i is not None}
                if role_start is not None else {})
    system_code = ROLE_CODES.index('system')
    open_id, close_id = engine.backbone.controls.open_id, engine.backbone.controls.close_id
    windows = {'train': [], 'test': []}
    masked_system_tokens = 0
    for row in rows:
        row_tokens = row['token_ids'] if 'token_ids' in row else engine._tokens(row['text'])
        labels = None
        context_tokens = 0
        if role_start is not None:
            labels = chat_roles([open_id] + list(row_tokens) + [close_id], start_id=role_start,
                                role_ids=role_ids, think_open=_special_id(tokenizer, '<think>'),
                                think_close=_special_id(tokenizer, '</think>'))
            if mask_system_prompt:
                index = 1
                while index < len(labels) and labels[index] == 0:
                    index += 1
                while index < len(labels) and labels[index] == system_code:
                    index += 1
                context_tokens = index - 1 if any(c == system_code for c in labels[1:index]) else 0
                masked_system_tokens += context_tokens
        for window in document_windows(row_tokens, open_id=open_id, close_id=close_id,
                                       tokens=tokens, prefix_tokens=prefix_tokens,
                                       supervised_suffix_start=row.get('supervised_suffix_start'),
                                       context_tokens=context_tokens):
            if labels is not None and row['split'] == 'test':
                window['roles'] = labels[window['start']:window['start'] + len(window['ids'])]
            windows[row['split']].append({**window,
                'document': hashlib.sha256(row['text'].encode()).hexdigest(),
                'groups': row['source_groups']})
    held, selection = select_held_document_windows(windows['test'], held_documents)
    group_order_sha256 = hashlib.sha256(json.dumps(
        selection['group_order'], ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    comparable_selection = {k: selection[k] for k in
                            ('policy', 'limit_documents', 'selected_documents', 'window_policy')}
    comparable_selection.update(group_count=len(selection['group_order']),
                                group_order_sha256=group_order_sha256)
    return held, receipt, comparable_selection, {
        'role_markers_available': role_start is not None,
        'mask_system_prompt': bool(mask_system_prompt),
        'masked_system_tokens': masked_system_tokens,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'records', 'pieces', 'text-data', 'match-report', 'match-plan', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--tokens', type=int, default=16384)
    parser.add_argument('--prefix-tokens', type=int, default=32)
    parser.add_argument('--held-documents', type=int, default=16)
    parser.add_argument('--mask-system-prompt', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError('fresh immutable diagnostic output required')
    if args.steps < 2:
        raise ValueError('autoregressive controls require at least two steps')
    match_plan = json.loads(args.match_plan.read_text())
    match_report = json.loads(args.match_report.read_text())
    options = match_plan.get('identity', {}).get('options', {})
    expected_inputs = match_plan.get('identity', {}).get('inputs', {})
    for name, path in (('records', args.records), ('pieces', args.pieces), ('text_data', args.text_data)):
        option_path = options.get(name)
        expected_sha = expected_inputs.get(option_path) if option_path else None
        if not expected_sha or sha(path) != expected_sha:
            raise ValueError(f'{name} does not match the named warm-up plan input')
    if options.get('tokens') != args.tokens or options.get('prefix_tokens') != args.prefix_tokens:
        raise ValueError('token window geometry differs from the named warm-up plan')
    if options.get('held_documents') != args.held_documents:
        raise ValueError('held document count differs from the named warm-up plan')
    if options.get('mask_system_prompt') is not args.mask_system_prompt:
        raise ValueError('system-prompt masking differs from the named warm-up plan')
    baseline_ar = match_report.get('autoregressive_controls')
    if not isinstance(baseline_ar, dict) or baseline_ar.get('steps') != args.steps:
        raise ValueError('autoregressive steps do not match the named warm-up control')
    if baseline_ar.get('windows') != 1:
        raise ValueError('the named warm-up control is not a single matched rollout window')
    if baseline_ar.get('start') != 'first assistant token':
        raise ValueError('named warm-up control did not start at the first assistant token')
    if match_report.get('step') != 32512:
        raise ValueError('the comparison report is not the qualified step-32512 warm-up report')

    checkpoint_sha = sha(args.checkpoint)
    torch.set_num_threads(2)
    engine, state = load_recurrence_checkpoint(args.checkpoint, device=args.device, dtype=torch.bfloat16)
    engine.backbone.eval()
    engine.heads.eval()
    held, text_receipt, selection, masking = _held_windows(
        engine, args.records, args.pieces, args.text_data,
        tokens=args.tokens, prefix_tokens=args.prefix_tokens,
        held_documents=args.held_documents,
        mask_system_prompt=args.mask_system_prompt)
    expected_selection = match_report.get('held_probe_selection')
    if selection != expected_selection:
        raise ValueError('held selection differs from the named warm-up report')
    if not held:
        raise ValueError('no held windows in the matched selection')

    # The warm-up control used the first window in its first equal-geometry
    # evaluation batch. Evaluate that same window first, then all other held
    # windows sequentially so GPU memory does not scale with the held count.
    eval_batch_size = int(options.get('eval_batch', 4))
    batches = list(evaluation_batches(held, eval_batch_size, max_tokens=args.tokens))
    if not batches or not batches[0]:
        raise ValueError('matched held selection produced no evaluation batch')
    first_window = batches[0][0]
    ordered = [first_window] + [w for w in held if w is not first_window]
    assistant_codes = {ROLE_CODES.index('assistant_reasoning'), ROLE_CODES.index('assistant_reply')}
    rows = []
    for window in ordered:
        if 'roles' not in window:
            rows.append({'document_sha256': window['document'], 'source_groups': window['groups'],
                         'offset': window['offset'], 'excluded': 'chat role markers unavailable'})
            continue
        first_assistant = next((i for i, code in enumerate(window['roles'])
                                if i >= 1 and code in assistant_codes), None)
        if first_assistant is None or len(window['ids']) - first_assistant < 16:
            rows.append({'document_sha256': window['document'], 'source_groups': window['groups'],
                         'offset': window['offset'], 'excluded': 'no sufficiently long assistant suffix'})
            continue
        ids = torch.tensor([window['ids']], device=args.device)
        prefix, span = ids[:, :first_assistant], ids[:, first_assistant:]
        with torch.no_grad():
            scores = autoregressive_history_metrics(engine.backbone, engine.heads, prefix, span, steps=args.steps)
        rows.append({'document_sha256': window['document'], 'source_groups': window['groups'],
                     'offset': window['offset'], 'assistant_start': first_assistant,
                     'span_tokens': int(span.shape[1]), 'scores': scores})
    if sha(args.checkpoint) != checkpoint_sha:
        raise ValueError('checkpoint changed while diagnostics were running')
    if not rows or rows[0].get('document_sha256') != first_window['document']:
        raise RuntimeError('the reproduced first warm-up evaluation window changed')
    args.out.mkdir(parents=True)
    report = {
        'schema': 'natlang.recurrence-held-autoregressive-history/1',
        'checkpoint_step': state['step'],
        'checkpoint_sha256': checkpoint_sha,
        'checkpoint_port_profile': engine.heads.profile,
        'inputs': {str(p): sha(p) for p in
                   (args.records, args.pieces, args.text_data, args.match_plan, args.match_report)},
        'warmup_control_step': match_report['step'],
        'warmup_control_ar_steps': baseline_ar['steps'],
        'held_selection': selection,
        'text_row_receipt': text_receipt,
        'system_prompt_policy': masking,
        'reconstructed_baseline_window': {
            'document_sha256': first_window['document'],
            'source_groups': first_window['groups'],
            'offset': first_window['offset'],
        },
        'baseline_ar_window_identity_recorded': False,
        'baseline_window_binding_method': 'same plan-bound text data, held selection, token geometry, equal-geometry evaluation batching, first batch window, and first assistant-token split; the warm-up report stored only the selection and not the rollout window identity',
        'rollout_controls': ['gold', 'ar_greedy', 'ar_projection', 'ar_sketch'],
        'rows': rows,
        'foundation_qualified': False,
        'runtime_qualified': False,
        'task_qualified': False,
        'scope': 'Matched held text history controls on exact recurrence weights. Uses shared autoregressive_history_metrics; not task success, autonomous stopping, or a runtime certificate.'
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'schema': report['schema'], 'checkpoint_step': report['checkpoint_step'],
                      'checkpoint_sha256': checkpoint_sha, 'selected_windows': len(rows),
                      'excluded_windows': sum('excluded' in row for row in rows),
                      'first_row_document_sha256': rows[0]['document_sha256'],
                      'scope': report['scope']}), flush=True)


if __name__ == '__main__':
    main()
