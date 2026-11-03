"""Shared contract for learned-operator case converters (plans/neuralese/COMPACTION_OPERATOR_CONVERTERS.md §0).

A converter turns one data source into `natlang.port-record/1` records for exactly one learned operator
(training/neuralese/natlang_neuralese/model/learned_operators.py). It never renders, tokenizes or admits rows; the
shared renderer (natlang_neuralese/data/operator_graph.py) does that and applies the current protection union.

    with CaseSink('compact-value', 'tool-digest', converter=__file__, inputs=[path]) as sink:
        for record in records:
            sink.offer(record)          # validates, dedupes, seeds, writes or records the reject reason

Seeds (owner correction 2026-11-03): each source's ``meta.operator_seed = {text, provenance, origin}`` starts the
operator's producer span. A converter supplies ``gold``/``checked`` seeds where its source has them (supporting
facts, reference summaries, answer sentences, code-derived recall); ``offer`` fills every other source with the one
shared deterministic mechanical seed (natlang_neuralese/data/operator_seeds.py ``seed_for``), so every written case
carries an explicit seed and the manifest counts seeds by provenance.
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
from neuralese_data.records import normalize_text, seal, validate  # noqa: E402
from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402  (records.py puts it on sys.path)

OUT_ROOT = Path(os.environ.get('NATLANG_OPERATOR_CASES') or data_path('data_hdd', 'natlang-compaction-operators-20261103', 'cases'))
SCHEMA = 'natlang.operator-cases/1'

# Size floors follow the runtime's own cut thresholds (ts-host native/cutoff.ts): only values the runtime would cut
# are compaction cases. Ceilings keep one rendered graph below the renderer's token cap.
RULES = {
    # compact-value has its own lower floor (owner 2026-11-03, converter rerun): the runtime cuts a tool output only past
    # 2,000 shown characters, but the operator's state is also what a consumer reads instead of a value (also between
    # runs and in listings), and real tool outputs worth carrying exactly are often 800-2,000 characters (a directory
    # listing, a short file, a search page). Below 2,000 the text-cut sibling is the whole value, so such a case teaches
    # exact-detail carriage; the ablation baseline there is the full text. Other operators keep their own floors.
    'compact-value': {'sources': (1, 1), 'roles': {'tool_output', 'file', 'document', 'passage', 'record', 'schema',
                                                   'tool_doc'}, 'min_chars': 800, 'max_chars': 120_000},
    # One site per argument the listing cuts (>= SCOPE_VALUE_CHARS 4096); shorter arguments of a multi-argument
    # call are shown in full. The total keeps one rendered graph below the renderer's 32k-token cap.
    'compact-params': {'sources': (1, 8), 'roles': {'document', 'passage', 'file', 'record', 'schema', 'tool_output'},
                       'min_chars': 200, 'max_chars': 120_000, 'compact_min_chars': 4096, 'max_total_chars': 110_000},
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
    lengths = []
    for source in sources:
        try:
            lengths.append(len(source_text(source)))
        except ValueError:
            pass
    if rules.get('compact_min_chars') and lengths and max(lengths) < rules['compact_min_chars']:
        problems.append(f'no source reaches the compaction floor {rules["compact_min_chars"]}')
    if rules.get('max_total_chars') and sum(lengths) > rules['max_total_chars']:
        problems.append(f'sources total {sum(lengths)} chars above {rules["max_total_chars"]}')
    if 'sources' not in (record.get('consumer') or {}).get('withheld', []):
        problems.append('consumer must withhold sources')
    if (record.get('outcome') or {}).get('label') not in IMITATION_LABELS:
        problems.append(f'outcome.label {(record.get("outcome") or {}).get("label")!r} is not an imitation label')
    if not record.get('split_groups'):
        problems.append('split_groups empty')
    from natlang_neuralese.data.operator_seeds import supplied_seed
    for index, source in enumerate(sources):
        try:
            supplied_seed(source)
        except ValueError as error:
            problems.append(f'sources[{index}]: {error}')
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


def question_selected_spans(text: str, question: str, answers, budget: int, *, limit: int = 8) -> str | None:
    """Checked extractive seed for a question-driven tool-output case: the sentences/lines that contain a gold answer
    (best question overlap first, at most ``limit``), then the best question-overlap sentences while budget remains,
    all in source order. None when no sentence contains an answer (the state could not carry it)."""
    from natlang_neuralese.data.operator_seeds import _words, sentences
    wanted = [answer.lower() for answer in answers if isinstance(answer, str) and answer.strip()]
    asked = _words(question)
    pieces = list(dict.fromkeys(sentences(text)))

    def window(piece):  # a single over-long line keeps the stretch around its first answer occurrence
        if len(piece) <= budget:
            return piece
        at = min((i for i in (piece.lower().find(answer) for answer in wanted) if i >= 0), default=0)
        start = max(0, min(at - budget // 2, len(piece) - budget))
        return piece[start:start + budget]

    pieces = [window(piece) for piece in pieces]
    with_answer = sorted((-len(_words(piece) & asked), index) for index, piece in enumerate(pieces)
                         if any(answer in piece.lower() for answer in wanted))
    if not with_answer:
        return None
    chosen, used = set(), 0
    for _, index in with_answer[:limit]:
        if chosen and used + len(pieces[index]) + 1 > budget:
            continue
        chosen.add(index)
        used += len(pieces[index]) + 1
    extra = sorted((-len(_words(piece) & asked), index) for index, piece in enumerate(pieces)
                   if index not in chosen and _words(piece) & asked)
    for _, index in extra:
        if len(chosen) < 2 * limit and used + len(pieces[index]) + 1 <= budget:
            chosen.add(index)
            used += len(pieces[index]) + 1
    return '\n'.join(pieces[index] for index in sorted(chosen))


def pool_text_hashes(operator: str, *, exclude=(), out_root: Path | None = None) -> set:
    """Normalized-text hashes of every source already written for ``operator`` (source files of the pool except the ``exclude`` names), so a
    new converter can reject a value the operator already trains on (repetition across converters, not only within one)."""
    hashes = set()
    for path in sorted((Path(out_root or OUT_ROOT) / operator).glob('*.port-records.jsonl')):
        if path.name.removesuffix('.port-records.jsonl') in exclude:
            continue
        for record in iter_jsonl(path):
            for source in record['sources']:
                hashes.add(hashlib.sha256(normalize_text(source_text(source)).encode()).hexdigest())
    return hashes


def text_key(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


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
        self.counts, self.rejects, self.seen, self.seeds = Counter(), Counter(), set(), Counter()

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
        from natlang_neuralese.data.operator_seeds import seed_for
        filled = False
        for source in record['sources']:
            seed = seed_for(self.operator, record, source)
            if (source.get('meta') or {}).get('operator_seed') is None:
                source.setdefault('meta', {})['operator_seed'] = seed
                filled = True
            self.seeds[seed['provenance']] += 1
        if filled:
            seal(record)  # the content hash covers sources, so a filled seed reseals the record
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
                    'seed_provenance': dict(self.seeds),
                    'notes': self.notes, 'seconds': round(time.time() - self.started, 1),
                    'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    'admission': 'operator-case; rendering, protection union and training admission are separate'}
        tmp = self.manifest_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(manifest, indent=1) + '\n')
        os.replace(tmp, self.manifest_path)
        return False
