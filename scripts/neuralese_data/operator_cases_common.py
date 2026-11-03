"""Shared contract for learned-operator case converters (plans/neuralese/COMPACTION_OPERATOR_CONVERTERS.md §0).

A converter turns one data source into `natlang.port-record/1` records for exactly one learned operator
(training/neuralese/natlang_neuralese/model/learned_operators.py). It never renders, tokenizes or admits rows; the
shared renderer (natlang_neuralese/data/operator_graph.py) does that and applies the current protection union.

    with CaseSink('compact-value', 'tool-digest', converter=__file__, inputs=[path]) as sink:
        for record in records:
            sink.offer(record)          # validates, dedupes, writes or records the reject reason
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuralese_data.records import normalize_text, validate  # noqa: E402
from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402  (records.py puts it on sys.path)

OUT_ROOT = Path(os.environ.get('NATLANG_OPERATOR_CASES') or data_path('data_hdd', 'natlang-compaction-operators-20261103', 'cases'))
SCHEMA = 'natlang.operator-cases/1'

# Size floors follow the runtime's own cut thresholds (ts-host native/cutoff.ts): only values the runtime would cut
# are compaction cases. Ceilings keep one rendered graph below the renderer's token cap.
RULES = {
    'compact-value': {'sources': (1, 1), 'roles': {'tool_output', 'file', 'document', 'passage', 'record', 'schema',
                                                   'tool_doc'}, 'min_chars': 2000, 'max_chars': 120_000},
    'compact-params': {'sources': (1, 1), 'roles': {'document', 'passage', 'file', 'record', 'schema', 'tool_output'},
                       'min_chars': 4096, 'max_chars': 120_000},
    'compact-history': {'sources': (1, 1), 'roles': {'trajectory'}, 'min_chars': 8000, 'max_chars': 200_000},
    'extract': {'sources': (2, 8), 'roles': {'document', 'passage', 'record', 'file', 'tool_output'},
                'min_chars': 200, 'max_chars': 20_000},
}
IMITATION_LABELS = {'gold', 'checked'}


def source_text(source: dict) -> str:
    """The exact text a site reads for one source (messages are serialized like port_record_graph does)."""
    if isinstance(source.get('text'), str):
        return source['text']
    if isinstance(source.get('messages'), list):
        return json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    raise ValueError('source needs exact text or a message list')


def operator_problems(record: dict, operator: str) -> list[str]:
    """Operator-specific requirements on top of the port-record schema."""
    rules = RULES[operator]
    problems = []
    sources = record.get('sources') or []
    low, high = rules['sources']
    if not low <= len(sources) <= high:
        problems.append(f'sources: {len(sources)} not in [{low}, {high}]')
    for index, source in enumerate(sources):
        if source.get('role') not in rules['roles']:
            problems.append(f'sources[{index}].role {source.get("role")!r} not allowed for {operator}')
        try:
            length = len(source_text(source))
        except ValueError as error:
            problems.append(f'sources[{index}]: {error}')
            continue
        if length < rules['min_chars']:
            problems.append(f'sources[{index}]: {length} chars below {rules["min_chars"]}')
        if length > rules['max_chars']:
            problems.append(f'sources[{index}]: {length} chars above {rules["max_chars"]}')
    if 'sources' not in (record.get('consumer') or {}).get('withheld', []):
        problems.append('consumer must withhold sources')
    if (record.get('outcome') or {}).get('label') not in IMITATION_LABELS:
        problems.append(f'outcome.label {(record.get("outcome") or {}).get("label")!r} is not an imitation label')
    if not record.get('split_groups'):
        problems.append('split_groups empty')
    return problems


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path, limit=None):
    """Stream JSON lines from a plain or .zst file."""
    path = Path(path)
    if path.suffix == '.zst':
        import zstandard
        stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, 'rb')), encoding='utf-8')
    else:
        stream = open(path, encoding='utf-8')
    with stream:
        for count, line in enumerate(stream):
            if limit is not None and count >= limit:
                return
            if line.strip():
                yield json.loads(line)


def stable_rank(key: str, salt: str = 'operator-cases') -> int:
    """Deterministic order for capped selection: sort by this, keep the first N."""
    return int(hashlib.sha256(f'{salt}\0{key}'.encode()).hexdigest()[:16], 16)


class CaseSink:
    """Writes one (operator, source) case file, its rejects and its manifest; dedupes by normalized source text."""

    def __init__(self, operator: str, source: str, *, converter: str, inputs, out_root: Path = OUT_ROOT,
                 caps: dict | None = None, notes: str = ''):
        if operator not in RULES:
            raise ValueError(f'unknown operator {operator!r}')
        self.operator, self.source = operator, source
        self.dir = Path(out_root) / operator
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f'{source}.port-records.jsonl'
        self.rejects_path = self.dir / f'{source}.rejects.jsonl'
        self.manifest_path = self.dir / f'{source}.manifest.json'
        self.converter = Path(converter).resolve()
        self.inputs = [Path(item) for item in inputs]
        self.caps = dict(caps or {})
        self.notes = notes
        self.counts, self.rejects, self.seen = Counter(), Counter(), set()

    def __enter__(self):
        self.tmp = self.path.with_suffix('.jsonl.tmp')
        self.out = open(self.tmp, 'w', encoding='utf-8')
        self.rej = open(self.rejects_path, 'w', encoding='utf-8')
        self.started = time.time()
        return self

    def reject(self, record_id, reason: str):
        self.rejects[reason.split(':')[0]] += 1
        self.rej.write(json.dumps({'id': record_id, 'reason': reason}, ensure_ascii=False) + '\n')

    def offer(self, record: dict) -> bool:
        problems = validate(record) + operator_problems(record, self.operator)
        if problems:
            self.reject(record.get('id'), 'invalid: ' + '; '.join(problems[:3]))
            return False
        key = hashlib.sha256('\0'.join(normalize_text(source_text(source)) for source in record['sources'])
                             .encode()).hexdigest()
        target_key = hashlib.sha256((key + json.dumps(record['target'], sort_keys=True)).encode()).hexdigest()
        if target_key in self.seen:
            self.reject(record['id'], 'duplicate: same sources and target')
            return False
        self.seen.add(target_key)
        split = 'test' if record['split'] in ('test', 'validation') else 'train'
        self.counts[split] += 1
        self.out.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
        return True

    def __exit__(self, kind, value, traceback):
        self.out.close()
        self.rej.close()
        if kind is not None:
            return False
        os.replace(self.tmp, self.path)
        from natlang_neuralese.model.learned_operators import operator as registered
        manifest = {'schema': SCHEMA, 'operator': self.operator, 'operator_version': registered(self.operator).version,
                    'source': self.source, 'converter': str(self.converter),
                    'converter_sha256': _sha256_file(self.converter),
                    'inputs': [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': _sha256_file(path)}
                               for path in self.inputs],
                    'output': {'path': str(self.path), 'sha256': _sha256_file(self.path),
                               'bytes': self.path.stat().st_size},
                    'counts': dict(self.counts), 'rejects': dict(self.rejects), 'caps': self.caps,
                    'notes': self.notes, 'seconds': round(time.time() - self.started, 1),
                    'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    'admission': 'operator-case; rendering, protection union and training admission are separate'}
        tmp = self.manifest_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(manifest, indent=1) + '\n')
        os.replace(tmp, self.manifest_path)
        return False
