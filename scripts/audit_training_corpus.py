#!/usr/bin/env python3
"""Token-exact, resumable admission audit between rendering and training.

Never truncates examples. Outputs a filtered SFT corpus, renderer-compatible
manifest, per-record rejection ledger, and measured token/concentration report.
No benchmark, model forward pass, or external code execution is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_training_corpus import (_atomic, _atomic_new, _file_sha, _jsonl_bytes,
                                    _sha, _tokenizer_info)


def evidence(row):
    """Admission, execution fidelity, and specification evidence are distinct."""
    explicit = row.get('behavioral_evidence') or row.get('evidence')
    if explicit:
        return explicit.get('kind', 'declared') if isinstance(explicit, dict) else str(explicit)
    if row.get('execution_verified'):
        return 'execution_verified_unspecified_oracle'
    if row.get('teacher_trajectory_id'):
        return 'accepted_teacher_trajectory'
    if row.get('training_admission', {}).get('approved'):
        return 'admitted_without_independent_behavioral_evidence'
    return 'unknown'


def assess(row, tokenizer, end_token, max_len):
    source = row.get('source') or {}
    source = source if isinstance(source, dict) else {'name': str(source)}
    summary = {'id': row.get('id'), 'source': source.get('name', 'unknown'),
               'repository': source.get('repository') or source.get('repo') or source.get('name', 'unknown'),
               'family': row.get('family', 'unknown'), 'evidence': evidence(row),
               'split': row.get('split', 'unspecified'),
               'implementation': row.get('implementation_sha256'),
               'template': (row.get('generation') or {}).get('family') or row.get('family', 'unknown'),
               'prompt_tokens': 0, 'supervised_tokens': 0, 'total_tokens': 0}
    prompt, target = row.get('prompt'), row.get('completion')
    reason = None
    if not isinstance(prompt, str) or not isinstance(target, str) or not prompt or not target:
        reason = 'empty_or_invalid_pair'
    elif row.get('training_admission', {}).get('approved') is not True:
        reason = 'not_explicitly_admitted'
    elif not target.endswith(end_token) or end_token in target[:-len(end_token)]:
        reason = 'invalid_target_termination'
    elif not target[:-len(end_token)].strip():
        reason = 'empty_assistant_target'
    elif summary['split'] not in ('train', 'test', 'validation', 'valid', 'dev'):
        reason = 'missing_or_invalid_split'
    if not reason:
        # Match train_lora.encode exactly, including no implicit BOS/EOS insertion.
        x = tokenizer(prompt, add_special_tokens=False)['input_ids']
        y = tokenizer(target, add_special_tokens=False)['input_ids']
        summary.update(prompt_tokens=len(x), supervised_tokens=len(y), total_tokens=len(x) + len(y))
        if not x or not y:
            reason = 'empty_tokenized_pair'
        elif tokenizer(prompt + target, add_special_tokens=False)['input_ids'] != x + y:
            reason = 'tokenization_boundary_mismatch'
        elif len(x) + len(y) > max_len:
            reason = 'over_token_budget'
    summary['reason'] = reason
    summary['pair_sha256'] = _sha(_jsonl_bytes([{'prompt': prompt, 'completion': target}]))
    return {'summary': summary, 'record': None if reason else {**row, 'token_counts': {
        key: summary[key] for key in ('prompt_tokens', 'supervised_tokens', 'total_tokens')}}}


def report_for(assessed, max_len):
    result = {'input_rows': len(assessed), 'usable_rows': 0, 'rejected_rows': 0,
              'max_len': max_len, 'rejections': {}, 'distributions': {},
              'interpretation': 'Supervised tokens include assistant termination. Evidence labels are not interchangeable; replay alone is not an independent specification oracle.'}
    rejected = Counter()
    usable = [item['summary'] for item in assessed if item['record'] is not None]
    result['usable_rows'] = len(usable)
    result['rejected_rows'] = len(assessed) - len(usable)
    for item in assessed:
        if item['summary']['reason']:
            rejected[item['summary']['reason']] += 1
    result['rejections'] = dict(sorted(rejected.items()))
    for axis in ('source', 'repository', 'family', 'template', 'evidence', 'split'):
        groups = {}
        for item in assessed:
            row = item['summary']
            bucket = groups.setdefault(str(row[axis]), {'input_rows': 0, 'usable_rows': 0,
                                                         'rejected_rows': 0, 'prompt_tokens': 0,
                                                         'supervised_tokens': 0, 'total_tokens': 0,
                                                         'train_rows': 0, 'train_supervised_tokens': 0})
            bucket['input_rows'] += 1
            bucket['rejected_rows' if row['reason'] else 'usable_rows'] += 1
            if not row['reason']:
                for key in ('prompt_tokens', 'supervised_tokens', 'total_tokens'):
                    bucket[key] += row[key]
                if row['split'] == 'train':
                    bucket['train_rows'] += 1
                    bucket['train_supervised_tokens'] += row['supervised_tokens']
        total = sum(b['supervised_tokens'] for b in groups.values())
        train_total = sum(b['train_supervised_tokens'] for b in groups.values())
        for bucket in groups.values():
            bucket['supervised_token_fraction'] = bucket['supervised_tokens'] / total if total else 0
            bucket['train_supervised_token_fraction'] = bucket['train_supervised_tokens'] / train_total if train_total else 0
        result['distributions'][axis] = dict(sorted(groups.items()))
    result['tokens'] = {key: sum(row[key] for row in usable)
                        for key in ('prompt_tokens', 'supervised_tokens', 'total_tokens')}
    result['training_tokens'] = {key: sum(row[key] for row in usable if row['split'] == 'train')
                                 for key in ('prompt_tokens', 'supervised_tokens', 'total_tokens')}
    lengths = sorted(row['total_tokens'] for row in usable)
    result['length_tokens'] = {name: lengths[min(len(lengths) - 1, int((len(lengths) - 1) * q))] if lengths else 0
                               for name, q in [('p50', .5), ('p95', .95), ('p99', .99), ('max', 1)]}
    result['duplicate_clusters'] = {}
    for key in ('pair_sha256', 'implementation'):
        counts = Counter(row[key] for row in usable if row.get(key))
        result['duplicate_clusters'][key] = {'clusters': sum(n > 1 for n in counts.values()),
            'excess_rows': sum(n - 1 for n in counts.values()),
            'largest': [{'sha256': k, 'rows': n} for k, n in counts.most_common(20) if n > 1]}
    result['ready'] = any(row['split'] == 'train' for row in usable)
    return result


def audit_corpus(source, output, *, model, revision=None, max_len=8192, chunk_rows=128,
                 tokenizer=None, should_stop=lambda: False, rejection_ledgers=()):
    source, output = Path(source), Path(output)
    if max_len < 2 or chunk_rows < 1:
        raise ValueError('max_len >= 2 and chunk_rows >= 1 are required')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    _, current = _tokenizer_info(tokenizer, model, revision)
    source_manifest_path = source.with_name(source.name + '.manifest.json')
    source_manifest = json.loads(source_manifest_path.read_text())
    renderer = source_manifest['renderer']
    for key in ('model', 'revision', 'template_sha256', 'tokenizer_fingerprint_sha256', 'local_tokenizer_artifacts_sha256'):
        if renderer.get(key) != current.get(key):
            raise ValueError(f'rendered corpus/tokenizer mismatch: {key}')
    if source_manifest['sha256'] != _file_sha(source):
        raise ValueError('rendered corpus hash mismatch')
    end_token = renderer.get('end_token')
    if not isinstance(end_token, str) or not end_token:
        raise ValueError('renderer must declare target end_token')
    identity = {'version': 'natlang.token_audit/1', 'source_sha256': _file_sha(source),
                'manifest_sha256': _file_sha(source_manifest_path), 'renderer': renderer,
                'max_len': max_len, 'chunk_rows': chunk_rows, 'auditor_sha256': _file_sha(Path(__file__)),
                'rejection_ledgers': {str(Path(path).resolve()): _file_sha(Path(path)) for path in rejection_ledgers}}
    identity_sha = _sha(json.dumps(identity, sort_keys=True).encode())
    cache = output.with_name(output.name + '.audit-cache')
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = cache / 'manifest.json'
        state = {'identity': identity, 'identity_sha256': identity_sha, 'chunks': []}
        if manifest_path.exists():
            state = json.loads(manifest_path.read_text())
            if state['identity_sha256'] != identity_sha:
                raise ValueError('audit identity changed; use a new output path')
        else:
            _atomic(manifest_path, (json.dumps(state, indent=2) + '\n').encode())
        # Resume only from committed chunks; orphan chunk files may be rebuilt.
        assessed, cursor = [], 0
        for index, chunk in enumerate(state['chunks']):
            if chunk['index'] != index or chunk['start'] != cursor or chunk['end'] <= cursor:
                raise ValueError('invalid audit chunk boundaries')
            payload = (cache / chunk['file']).read_bytes()
            if _sha(payload) != chunk['sha256']:
                raise ValueError('audit cache corruption')
            loaded = [json.loads(line) for line in payload.splitlines()]
            if len(loaded) != chunk['end'] - cursor:
                raise ValueError('audit chunk row count mismatch')
            assessed.extend(loaded)
            cursor = chunk['end']
        def commit(batch, start):
            selected = [assess(row, tokenizer, end_token, max_len) for row in batch]
            index = len(state['chunks'])
            name = f'chunk-{index:08d}.jsonl'
            payload = _jsonl_bytes(selected)
            _atomic(cache / name, payload)
            state['chunks'].append({'index': index, 'start': start, 'end': start + len(batch),
                                    'file': name, 'sha256': _sha(payload)})
            _atomic(manifest_path, (json.dumps(state, indent=2) + '\n').encode())
            assessed.extend(selected)
        batch, count = [], 0
        with source.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                count += 1
                if count <= cursor:
                    continue
                batch.append(json.loads(line))
                if len(batch) >= chunk_rows:
                    commit(batch, count - len(batch))
                    batch = []
                    if should_stop():
                        return 75
        if cursor > count or source_manifest['rows'] != count:
            raise ValueError('source row count mismatch')
        if batch:
            commit(batch, count - len(batch))
            if should_stop():
                return 75
        if _file_sha(source) != identity['source_sha256'] or _file_sha(source_manifest_path) != identity['manifest_sha256']:
            raise ValueError('source changed during audit')
        report = report_for(assessed, max_len)
        report['renderer'] = renderer
        report['upstream_render_rejections'] = source_manifest.get('rejections', {})
        report['upstream_rejection_ledgers'] = {}
        for path, checksum in identity['rejection_ledgers'].items():
            reasons = Counter()
            with Path(path).open() as stream:
                for line in stream:
                    if line.strip():
                        rejected_row = json.loads(line)
                        reasons[str(rejected_row.get('reason') or rejected_row.get('rejection') or rejected_row.get('error') or 'unspecified')] += 1
            if _file_sha(Path(path)) != checksum:
                raise ValueError('rejection ledger changed during audit')
            report['upstream_rejection_ledgers'][path] = {'sha256': checksum, 'rows': sum(reasons.values()),
                'reasons': dict(sorted(reasons.items())),
                'scope': 'Upstream-stage events, not additive to this rendered corpus rejection count.'}
        records = [item['record'] for item in assessed if item['record'] is not None]
        rejects = [item['summary'] for item in assessed if item['record'] is None]
        data = _jsonl_bytes(records)
        report_bytes = (json.dumps(report, indent=2, sort_keys=True) + '\n').encode()
        rejection_bytes = _jsonl_bytes(rejects)
        final = {**source_manifest, 'source': [str(source.resolve())],
                 'source_sha256': [identity['source_sha256']], 'rows': len(records), 'sha256': _sha(data),
                 'identity_sha256': identity_sha, 'audit': {'max_len': max_len, 'ready': report['ready'],
                 'report_sha256': _sha(report_bytes), 'rejections_sha256': _sha(rejection_bytes)}}
        outputs = [(output, data), (output.with_name(output.name + '.audit.json'), report_bytes),
                   (output.with_name(output.name + '.rejected.jsonl'), rejection_bytes),
                   (output.with_name(output.name + '.manifest.json'), (json.dumps(final, indent=2) + '\n').encode())]
        # Commit manifest last. Identical orphan outputs are recoverable; differing files are never overwritten.
        for path, payload in outputs:
            if path.exists():
                if path.read_bytes() != payload:
                    raise ValueError(f'audit output corruption: {path}')
            else:
                _atomic_new(path, payload)
        return 0 if report['ready'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision')
    parser.add_argument('--max-len', type=int, default=8192)
    parser.add_argument('--chunk-rows', type=int, default=128)
    parser.add_argument('--rejection-ledger', action='append', type=Path, default=[])
    args = parser.parse_args()
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    return audit_corpus(args.input, args.output, model=args.model, revision=args.revision,
                        max_len=args.max_len, chunk_rows=args.chunk_rows, should_stop=lambda: stopping,
                        rejection_ledgers=args.rejection_ledger)


if __name__ == '__main__':
    sys.exit(main())
