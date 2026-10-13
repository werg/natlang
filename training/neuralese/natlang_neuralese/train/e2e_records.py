"""Streamed trajectory source for the end-to-end trainer (train/trajectories.py --record-source trajectories).

A whole e2e cohort (neuralese-e2e-cohort-20261012-v2: 174,608 records, 11 GB) does not fit in memory as parsed
records, and authenticating it as one file would require writer names to be unique across the whole corpus. This
module streams it instead, one trajectory (chain) at a time:

1. Index. One pass over each cohort's ``records.jsonl`` records every record's byte offset, split, family, target and
   admission, and the soft parts it reads. ``chains.jsonl`` (written by the cohort builder) groups records into
   trajectories. The index is cached under ``~/.cache/natlang/e2e-record-index``, keyed by the files' path, size and
   mtime, so later runs and resumes skip the pass.
2. Sampling. The direct trainer's sampler (text_supervision.CohortDocumentSampler) picks cohort = task family, then
   sampling group = the chain's first source group, then document = chain, then item = a trainable record of the
   chain. Families are drawn by their trainable-record count raised to ``family_temperature`` (0.5 by default:
   musique_hops, 42% of the trainable records and 72% of the edges, gets about 9% of the draws), so no one family
   dominates. Every draw uses a generator seeded by the run seed and the record cursor, so resume and step rollback
   replay the same records without extra checkpoint state.
3. Loading. The trainer reads a drawn chain's records by offset and authenticates the chain on its own
   (authenticated_recurrence_context_view per trajectory). Writer-name uniqueness is therefore checked per trajectory,
   and between the trajectories resident at the same time.

The held and probe chains are chosen per family by a seeded order and loaded once at start.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

INDEX_SCHEMA = 'natlang.e2e-record-index/1'


def cache_root() -> Path:
    return Path(os.environ.get('NATLANG_E2E_INDEX_CACHE') or Path.home() / '.cache' / 'natlang' / 'e2e-record-index')


def _file_key(path: Path) -> str:
    stat = path.stat()
    return f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'


def _atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def cohort_files(value) -> tuple[Path, Path]:
    """``(records.jsonl, chains.jsonl)`` of a cohort named by its directory or by its records file."""
    path = Path(value)
    records = path / 'records.jsonl' if path.is_dir() else path
    chains = records.parent / 'chains.jsonl'
    if not records.is_file() or not chains.is_file():
        raise ValueError(f'--record-source trajectories needs a cohort directory (or its records.jsonl) with '
                         f'chains.jsonl beside records.jsonl: {value}')
    return records, chains


def _soft_names(record) -> list[str]:
    names = []
    for message in record.get('messages') or []:
        content = message.get('content')
        if isinstance(content, list):
            names += [part['name'] for part in content if isinstance(part, dict) and part.get('type') == 'soft']
    if isinstance(record.get('target'), dict) and isinstance(record['target'].get('content'), list):
        names += [part['name'] for part in record['target']['content']
                  if isinstance(part, dict) and part.get('type') == 'soft']
    return names


def _declared_unadmitted(record) -> bool:
    admission = record.get('training_admission')
    return isinstance(admission, dict) and admission.get('approved') is not True


def build_index(value) -> dict:
    """The (cached) index of one cohort: ``records`` {id: [offset, length, split, family, trainable]}, ``chains``
    (chains.jsonl rows plus ``group``) and ``soft`` {name: first record id that reads it}."""
    from .trajectories import record_family
    records_path, chains_path = cohort_files(value)
    key = hashlib.sha256((INDEX_SCHEMA + _file_key(records_path) + _file_key(chains_path)).encode()).hexdigest()
    cached = cache_root() / f'{key}.json'
    if cached.exists():
        return json.loads(cached.read_text())
    records, soft, groups = {}, {}, {}
    offset = 0
    with records_path.open('rb') as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                rid = record['id']
                if rid in records:
                    raise ValueError(f'duplicate record id {rid} in {records_path}')
                trainable = bool(record.get('target')) and not _declared_unadmitted(record)
                records[rid] = [offset, len(line), record.get('split'), record_family(record), trainable]
                source_groups = sorted(g for g in (record.get('source_groups') or []) if isinstance(g, str) and g)
                if source_groups:
                    groups[rid] = source_groups[0]
                for name in _soft_names(record):
                    soft.setdefault(name, rid)
            offset += len(line)
    chains = []
    with chains_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            chain = json.loads(line)
            missing = [rid for rid in chain['records'] if rid not in records]
            if missing:
                raise ValueError(f'chain {chain["trajectory"]} names records missing from {records_path}: {missing[:3]}')
            chain['group'] = next((groups[rid] for rid in chain['records'] if rid in groups), chain['trajectory'])
            chains.append(chain)
    index = {'schema': INDEX_SCHEMA, 'records_path': str(records_path.resolve()),
             'chains_path': str(chains_path.resolve()), 'records': records, 'chains': chains, 'soft': soft}
    _atomic_write(cached, json.dumps(index, separators=(',', ':')))
    return index


class TrajectoryRecordSource:
    """Every trajectory of one or more cohorts, indexed on disk and loaded per chain."""

    def __init__(self, values, *, seed, family_temperature=0.5):
        from .text_supervision import CohortDocumentSampler
        if not family_temperature >= 0:
            raise ValueError('--family-temperature must be at least 0')
        self.seed, self.family_temperature = seed, family_temperature
        self.indexes = [build_index(value) for value in values]
        self.chains = []
        for position, index in enumerate(self.indexes):
            for chain in index['chains']:
                self.chains.append({**chain, 'cohort': position,
                                    'key': f'{position}:{chain["trajectory"]}' if len(self.indexes) > 1
                                    else chain['trajectory']})
        self.items = []  # (chain position, record id) of every trainable train-split record
        for position, chain in enumerate(self.chains):
            if chain['split'] != 'train':
                continue
            entries = self.indexes[chain['cohort']]['records']
            self.items += [(position, rid) for rid in chain['records']
                           if entries[rid][2] == 'train' and entries[rid][4]]
        if not self.items:
            raise ValueError('the streamed cohort has no trainable train-split records')
        counts = {}
        for position, _ in self.items:
            family = self.chains[position]['family']
            counts[family] = counts.get(family, 0) + 1
        raw = {family: count ** family_temperature for family, count in counts.items()}
        total = sum(raw.values())
        self.family_weights = {family: value / total for family, value in raw.items()}
        self.family_records = counts
        self.sampler = CohortDocumentSampler(
            self.items, self.family_weights,
            cohort_key=lambda item: self.chains[item[0]]['family'],
            document_key=lambda item: self.chains[item[0]]['key'],
            sampling_group_key=lambda item: f'{self.chains[item[0]]["family"]}|{self.chains[item[0]]["group"]}')
        self._streams = {}

    def soft_names(self) -> dict[str, tuple[int, str]]:
        """Every soft part any record reads: {name: (cohort, first record id reading it)}."""
        names = {}
        for position, index in enumerate(self.indexes):
            for name, rid in index['soft'].items():
                names.setdefault(name, (position, rid))
        return names

    def draw(self, cursor: int, attempt: int = 0) -> tuple[int, str]:
        """``(chain position, record id)`` of the record at ``cursor`` (a redraw passes the attempt number)."""
        rng = random.Random(f'{self.seed}:e2e-trajectory-stream:{cursor}:{attempt}')
        return self.sampler.sample(rng)

    def chosen_chains(self, split: str, per_family: int, label: str) -> list[int]:
        """Up to ``per_family`` chains of each family in ``split``, by a seeded order (the held and probe sets)."""
        by_family = {}
        for position, chain in enumerate(self.chains):
            if chain['split'] == split:
                by_family.setdefault(chain['family'], []).append(position)
        chosen = []
        for family in sorted(by_family):
            positions = sorted(by_family[family], key=lambda p: self.chains[p]['key'])
            random.Random(f'{self.seed}:e2e-{label}:{family}').shuffle(positions)
            chosen += positions[:per_family]
        return chosen

    def load_record(self, cohort: int, rid: str) -> dict:
        offset, length = self.indexes[cohort]['records'][rid][:2]
        stream = self._streams.get(cohort)
        if stream is None:
            stream = self._streams[cohort] = open(self.indexes[cohort]['records_path'], 'rb')
        stream.seek(offset)
        record = json.loads(stream.read(length))
        if record.get('id') != rid:
            raise ValueError(f'record index is stale: offset of {rid} holds {record.get("id")}; '
                             f'delete the cached index under {cache_root()}')
        return record

    def load_chain(self, position: int) -> list[dict]:
        chain = self.chains[position]
        return [self.load_record(chain['cohort'], rid) for rid in chain['records']]

    def train_records(self) -> int:
        return len(self.items)

    def receipt(self) -> dict:
        return {'policy': 'family-then-source-group-then-chain-then-record/1',
                'cohorts': [{'records': index['records_path'], 'chains_file': index['chains_path'],
                             'records_indexed': len(index['records']), 'chains': len(index['chains'])}
                            for index in self.indexes],
                'trainable_train_records': len(self.items),
                'family_temperature': self.family_temperature,
                'family_records': dict(sorted(self.family_records.items())),
                'family_weights': {family: round(weight, 6) for family, weight in sorted(self.family_weights.items())},
                'sampler': self.sampler.receipt(),
                'draw': 'random.Random(f"{seed}:e2e-trajectory-stream:{cursor}:{attempt}"), with replacement',
                'admission_granted': False}


def steps_for_epochs(source: TrajectoryRecordSource, epochs: float, batch: int) -> int:
    return max(1, math.ceil(epochs * source.train_records() / batch))
