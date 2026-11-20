#!/usr/bin/env python3
"""Teacher-model provenance of the SFT teacher data: which model and provider wrote each teacher episode, per source.

One row per teacher episode (or October trajectory) in OUT/episodes.jsonl.zst: source, key, model_id, provider (the
transport that served it: a natlang Pi provider such as `anthropic` / `openai-codex`, else derived from the HTTP
endpoint by `teacher_gen.endpoint_provider`, else the source's recorded serving), endpoint, reasoning_effort, origin,
calls, and the reasoning_content length of every call. OUT/summary.json aggregates per (source, model): episodes,
calls, calls with reasoning, reasoning chars p50/p90. The result is registered (`teacher-provenance-*`, kind
`trajectories`, inputs = the scanned datasets) so the per-model audit of teacher reasoning reads one file.

Sources:
  --pool DIR            a teacher_gen WORK: manifest.jsonl, every scored episode (success or not; no call contents)
  --episodes FILE       an sft_sources WORK's episodes*.jsonl.zst: the extracted teacher successes (source=teacher)
  --october DIR         sft_sources --october rows (GROUP.jsonl.zst, one row per model call; provenance.model)
  --native GLOB|@LIST   October `natlang.teacher_trajectory.native/1` files (Qwen3.6 / Qwen3.8 continuation batches)
"""
import argparse
import collections
import glob
import io
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

from teacher_gen import endpoint_provider  # noqa: E402

# models an October/acquired source names only per store (no per-row model field): the store's own model card
STORE_MODELS = {
    'OpenResearcher/OpenResearcher-Dataset': ('gpt-oss (OpenResearcher harmony)', 'upstream-dataset'),
    'CooperBench/qwen9b-coop-mini-swe-agent': ('qwen 9B (CooperBench qwen9b-coop)', 'upstream-dataset'),
    'CooperBench/team-coop': (None, 'upstream-dataset'),   # Codex + Qwen sessions; the per-session model is not recorded
}
# self-hosted October teachers: model id -> how it was served
SELF_HOSTED = {
    'nvidia/Qwen3.6-35B-A3B-NVFP4': 'self-hosted-vllm-dgx',
    'qwen3.6-35b-a3b': 'self-hosted-vllm',
    'Ternary-Bonsai-2-27B': 'self-hosted',
}


def zstd_lines(path):
    import zstandard
    with open(path, 'rb') as raw:
        yield from io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding='utf-8')


def lines(path):
    if str(path).endswith('.zst'):
        yield from zstd_lines(path)
    else:
        with open(path) as stream:
            yield from stream


def teacher_fields(teacher):
    """model_id, provider, endpoint, reasoning_effort of a teacher_gen manifest `teacher` record (any vintage)."""
    teacher = teacher or {}
    endpoint = teacher.get('endpoint')
    provider = teacher.get('provider') or (endpoint_provider(endpoint) if endpoint else None)
    return {'model_id': teacher.get('model_id'), 'provider': provider, 'endpoint': endpoint,
            'reasoning_effort': teacher.get('reasoning_effort')}


def reasoning_chars(target):
    return len((target or {}).get('reasoning_content') or '')


def scan_pool(pool, out):
    label = 'teacher:' + '/'.join(Path(pool).parts[-2:])
    for line in open(Path(pool) / 'manifest.jsonl'):
        if not line.strip():
            continue
        record = json.loads(line)
        group = Path(record['rollouts']).parent.name
        for e in record.get('episodes') or []:
            yield {'source': label, 'scope': 'pool', 'key': f"{label}:{group}:{e['sample']}", 'task_id': record['task_id'],
                   **teacher_fields(record.get('teacher')), 'outcome': e.get('outcome'),
                   'success': e.get('outcome') == 'task_success' and (e.get('reward') or 0) >= 1,
                   'completion_tokens': e.get('completion_tokens'), 'calls': e.get('turns')}


def scan_episodes(path, out):
    for line in zstd_lines(path):
        row = json.loads(line)
        if row.get('source') != 'teacher':
            continue
        chars = [reasoning_chars(t.get('target')) for t in row.get('turns') or []]
        yield {'source': row['run'], 'scope': 'sft-episodes', 'file': str(path), 'work': '/'.join(Path(path).parts[-2:]),
               'key': row['key'],
               'task_id': row.get('task_id'), **teacher_fields(row.get('teacher')), 'success': True,
               'calls': len(chars), 'reasoning_chars': chars,
               'tools': sorted({c['function']['name'] for t in row.get('turns') or []
                                for c in (t.get('target') or {}).get('tool_calls') or []})}


def october_model(group, provenance):
    model = provenance.get('model') or provenance.get('teacher')
    if model:
        return model, SELF_HOSTED.get(model) or ('openai-codex' if model.startswith('gpt-6') else
                                                 'upstream-dataset' if group.startswith('acq-') else None)
    store = provenance.get('store')
    if store in STORE_MODELS:
        return STORE_MODELS[store]
    return None, None


def scan_october(root, out):
    for path in sorted(Path(root).glob('*.jsonl.zst')):
        group = path.name[:-len('.jsonl.zst')]
        trajectories = {}
        for line in zstd_lines(path):
            row = json.loads(line)
            provenance = row.get('provenance') or {}
            key = f"october:{group}:{provenance.get('trajectory') or row.get('task_id')}"
            t = trajectories.get(key)
            if t is None:
                model, provider = october_model(group, provenance)
                outcome = provenance.get('outcome') or {}
                t = trajectories[key] = {
                    'source': 'october:' + group, 'scope': 'october', 'key': key, 'task_id': row.get('task_id'),
                    'model_id': model, 'provider': provider, 'endpoint': None, 'reasoning_effort': None,
                    'origin': provenance.get('source_file') or provenance.get('store'),
                    'store': provenance.get('store'), 'collection_role': provenance.get('collection_role'),
                    'success': bool(outcome.get('accepted') or outcome.get('resolved')) or None,
                    'calls': 0, 'reasoning_chars': [], 'tools': set(), 'rerendered': row.get('rerendered')}
            t['calls'] += 1
            t['reasoning_chars'].append(reasoning_chars(row.get('target')))
            t['tools'].update(c['function']['name'] for c in (row.get('target') or {}).get('tool_calls') or [])
        for t in trajectories.values():
            t['tools'] = sorted(t['tools'])
            yield t


def native_family(path):
    """The campaign a native-trajectories file belongs to: the directory above admission/, else the registered package
    without its batch/tranche suffix."""
    parts = Path(path).parts
    if 'admission' in parts:
        return parts[parts.index('admission') - 1]
    return re.sub(r'-(batch|tranche)\d+.*', '', parts[-3])


def scan_native(pattern, out):
    """October natlang.teacher_trajectory.native/1 files: one verified teacher trajectory per line."""
    paths = open(pattern[1:]).read().split() if pattern.startswith('@') else sorted(glob.glob(pattern))
    for path in paths:
        for line in lines(path):
            if not line.strip():
                continue
            row = json.loads(line)
            provenance = row.get('provenance') or {}
            turns = row.get('trajectory') or []
            chars, tools = [], set()
            for turn in turns:
                assistant = turn.get('assistant') or {}
                chars.append(len(assistant.get('reasoning') or assistant.get('reasoning_content') or ''))
                for call in assistant.get('calls') or assistant.get('tool_calls') or []:
                    tools.add(call.get('tool') or (call.get('function') or call).get('name'))
            outcome = row.get('outcome') or {}
            model = provenance.get('model')
            yield {'source': 'native:' + native_family(path), 'scope': 'native',
                   'file': path, 'key': row.get('id'), 'model_id': model,
                   'provider': SELF_HOSTED.get(model, 'self-hosted'), 'endpoint': None, 'reasoning_effort': None,
                   'origin': path, 'tool_schema': provenance.get('tool_schema'),
                   'runtime_contract_version': provenance.get('runtime_contract_version'),
                   'success': bool(outcome.get('accepted')),
                   'calls': len(turns), 'reasoning_chars': chars, 'tools': sorted(t for t in tools if t)}


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summarize(rows):
    table = collections.defaultdict(lambda: {'episodes': 0, 'successes': 0, 'calls': 0, 'chars': [],
                                             'providers': collections.Counter(), 'tools': collections.Counter(),
                                             'missing_model': 0, 'missing_provider': 0})
    for r in rows:
        t = table[(r.get('work') or '', r['source'], r.get('model_id') or 'unknown')]
        t['episodes'] += 1
        t['successes'] += bool(r.get('success'))
        t['calls'] += r.get('calls') or 0
        t['chars'].extend(r.get('reasoning_chars') or [])
        t['providers'][r.get('provider') or 'unknown'] += 1
        t['tools'].update(r.get('tools') or [])
        t['missing_model'] += not r.get('model_id')
        t['missing_provider'] += not r.get('provider')
    out = []
    for (work, source, model), t in sorted(table.items()):
        nonzero = [c for c in t['chars'] if c]
        out.append({**({'work': work} if work else {}), 'source': source, 'model_id': model, 'providers': dict(t['providers']), 'episodes': t['episodes'],
                    'successes': t['successes'], 'calls': t['calls'], 'calls_scanned': len(t['chars']),
                    'calls_with_reasoning': len(nonzero),
                    'reasoning_chars_p50_all_calls': percentile(t['chars'], .5),
                    'reasoning_chars_p50': percentile(nonzero, .5), 'reasoning_chars_p90': percentile(nonzero, .9),
                    'reasoning_chars_mean': round(statistics.fmean(nonzero)) if nonzero else None,
                    'missing_model': t['missing_model'], 'missing_provider': t['missing_provider'],
                    'top_tools': dict(t['tools'].most_common(8))})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--pool', action='append', default=[])
    p.add_argument('--episodes', action='append', default=[])
    p.add_argument('--october', action='append', default=[])
    p.add_argument('--native', action='append', default=[], help='glob of native-trajectories.jsonl files, or @FILE listing them')
    p.add_argument('--register', help='registry dataset id for OUT (inputs: the scanned paths)')
    args = p.parse_args(argv)
    import zstandard
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    scans = [(scan_pool, x) for x in args.pool] + [(scan_episodes, x) for x in args.episodes] + \
            [(scan_october, x) for x in args.october] + [(scan_native, x) for x in args.native]
    with open(out / 'episodes.jsonl.zst', 'wb') as raw:
        with zstandard.ZstdCompressor(level=10).stream_writer(raw) as writer:
            for scan, source in scans:
                n = 0
                for row in scan(source, out):
                    writer.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
                    rows.append({k: row.get(k) for k in ('work', 'source', 'model_id', 'provider', 'success', 'calls',
                                                         'reasoning_chars', 'tools')})
                    n += 1
                print(json.dumps({'event': 'scanned', 'scan': scan.__name__, 'source': source, 'rows': n}), flush=True)
    summary = {'schema': 'natlang.teacher-provenance-summary/1', 'inputs': {'pool': args.pool, 'episodes': args.episodes,
               'october': args.october, 'native': args.native}, 'by_source_model': summarize(rows)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    if args.register:
        from natlang_neuralese.common import data_registry
        inputs = [*args.pool, *args.episodes, *args.october]
        data_registry.record_dataset(
            args.register, out, kind='trajectories', include=['episodes.jsonl.zst', 'summary.json'],
            format='natlang.teacher-provenance/1: episodes.jsonl.zst (one teacher episode or October trajectory: source, '
                   'key, model_id, provider, endpoint, reasoning_effort, origin, calls, reasoning_chars per call, tools), '
                   'summary.json (per source and model)',
            rows=len(rows), inputs=inputs, license='metadata of the inputs (each keeps its registered license)',
            note='teacher-model provenance of the SFT teacher sources (per-model audit of teacher reasoning)',
            extra={'teacher_models': [{k: s.get(k) for k in ('work', 'source', 'model_id', 'providers', 'episodes', 'successes')}
                                      for s in summary['by_source_model']]})
    return 0


if __name__ == '__main__':
    sys.exit(main())
