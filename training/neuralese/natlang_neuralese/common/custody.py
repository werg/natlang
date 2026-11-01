"""World-split custody: the one mechanism that decides which worlds may teach, select or confirm.

Shared by the whole-graph RL learner and JOLT+ privileged context, the memetic learner's development and
confirmation slices, the tuning supply, the RL training curriculum and the decontamination verifiers.
Decomposition: ``DECOMPOSITION.md`` (this directory), part 1.

Units (all crisp: exact verifiers of what the stages produce):

1. ``world_identities`` is everything that relates two worlds: ids and source ids, source/split/pair
   groups, world bindings (service worlds, inputs and, between invocations being split, single evidence
   items) and two content fingerprints (the rendered world and the visible world). Two cases sharing any
   identity are related and stay on one side. Protected-world custody leaves out single evidence items:
   procedural generators draw them from shared banks ("no sun", "heater works"), and on the 20,640-row
   RL curriculum they alone related 7,845 rows (38%) to held packets whose worlds differ. A whole shared
   evidence set is still caught by the rendered-world fingerprint.
2. ``ProtectedWorlds`` holds the identities of every declared protected source in
   ``training/protected-worlds.json``: registered evaluation-only packets, held and quality source groups,
   decision holds and explicit test reservations. It fails closed when a declared source is unreadable or
   differs from its pinned digest. ``check`` returns the train-world receipt or raises ``CustodyError``.
3. ``related_components`` groups related worlds (union-find over shared identities). A split policy assigns
   whole components, never single rows.
4. ``verify_slices`` checks named slices (train/development/confirmation) for pairwise unrelatedness and,
   given protected worlds, refuses any protected-related row. ``check_slice_receipt`` re-verifies the bound
   receipt where the protected sources are not present (e.g. an evaluator sandbox).
"""
from __future__ import annotations

import functools
import json
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

from .hashing import canonical_json_sha256_hex, sha256_file_hex, sha256_hex

REGISTRY = 'training/protected-worlds.json'
REGISTRY_SCHEMA = 'natlang.protected-worlds-registry/1'
CUSTODY_SCHEMA = 'natlang.jolt-privileged-custody/1'  # train-world receipt, recorded in captures: keep stable
SLICES_SCHEMA = 'natlang.world-slices/1'
PROTECTED_ADMISSION = 'evaluation-only-never-training'
GOLD_FIELDS = ('expected', 'expected_files', 'oracle', 'files_oracle')


class CustodyError(ValueError):
    """A world would teach, select or confirm while related to a protected or other-side world."""


def _repo():
    return Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------- identities

def world_fingerprint(case):
    """Content identity of a case's rendered world, independent of its ids and split labels."""
    semantics = case.get('semantics') or {}
    curriculum = case.get('curriculum') or {}
    world = {key: semantics.get(key) for key in ('root', 'files', 'folder_files', 'services', 'inputs')}
    world['evidence'] = curriculum.get('evidence')
    return canonical_json_sha256_hex(world)


def visible_world_key(row):
    """Family plus every semantics field except gold expectations and oracles (folder_files, operation...)."""
    semantics = row.get('semantics', {})
    visible = {key: value for key, value in semantics.items() if key not in GOLD_FIELDS}
    return json.dumps([row['family'], visible], sort_keys=True, separators=(',', ':'))


def world_bindings(row):
    """Source, evidence, service-world and input bindings of one invocation (sorted strings)."""
    bindings = {'source:' + value for field in ('source_groups', 'source_ids')
                for value in row.get(field) or [] if isinstance(value, str) and value}
    evidence = (row.get('curriculum') or {}).get('evidence') or {}
    for field in ('world', 'retrieved', 'background'):
        for item in evidence.get(field) or []:
            bindings.add('evidence:' + canonical_json_sha256_hex(item))
    semantics = row.get('semantics') or {}
    if semantics.get('services'):
        bindings.add('service-world:' + canonical_json_sha256_hex(semantics['services']))
    if 'inputs' in semantics and semantics['inputs'] not in ({}, []):
        bindings.add('inputs:' + canonical_json_sha256_hex(semantics['inputs']))
    return sorted(bindings)


def world_identities(case, *, evidence_items=True):
    """Every identity relating this case's world to another: ``{(kind, value)}``.

    ``evidence_items=False`` (protected-world custody) omits single shared evidence items."""
    curriculum = case.get('curriculum') or {}
    ids = {('id', case.get('id'))}
    ids.update(('id', value) for value in case.get('source_ids') or [])
    ids.update(('group', value) for value in case.get('source_groups') or [])
    ids.update(('group', value) for value in case.get('split_groups') or [])
    for key in ('split_group', 'pair_group'):
        if curriculum.get(key):
            ids.add(('group', curriculum[key]))
    bindings = [*world_bindings(case), *((case.get('memetic_source') or {}).get('world_bindings') or [])]
    ids.update(('binding', value) for value in bindings if evidence_items or not value.startswith('evidence:'))
    ids.add(('world', world_fingerprint(case)))
    if isinstance(case.get('family'), str) and isinstance(case.get('semantics'), dict):
        ids.add(('visible', sha256_hex(visible_world_key(case).encode())))
    return {item for item in ids if isinstance(item[1], str) and item[1]}


# ---------------------------------------------------------------- protected worlds

def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _source_path(value, root):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _evaluation_packets(source, root):
    registry = _read_protected_registry(_source_path(source['registry'], root))
    entries = registry.get('corpora', registry) if isinstance(registry, dict) else registry
    identities, packets = set(), []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get('admission') != PROTECTED_ADMISSION:
            continue
        if 'cases.jsonl' not in (entry.get('include') or []):
            continue
        path = _source_path(entry['path'], root) / 'cases.jsonl'
        if not path.is_file():
            raise CustodyError(f'protected packet {entry["id"]} is not readable; custody fails closed')
        rows = _jsonl(path)
        for row in rows:
            identities |= world_identities(row, evidence_items=False)
        packets.append({'id': entry['id'], 'cases_sha256': sha256_file_hex(path), 'rows': len(rows)})
    if not packets:
        raise CustodyError('no protected evaluation packets were registered; custody fails closed')
    return identities, {'packets': packets}


def _source_groups(source, root):
    rows = _jsonl(_source_path(source['path'], root))
    return {('group', group) for row in rows for group in row.get('source_groups') or []}, {}


def _decision_holds(source, root):
    directory = _source_path(source['path'], root)
    files = sorted(directory.glob('*.jsonl'))
    if not files:
        raise CustodyError(f'decision holds {directory} are not readable; custody fails closed')
    ids = {('id', row['id']) for path in files for row in _jsonl(path) if isinstance(row.get('id'), str)}
    return ids, {'files': {path.name: sha256_file_hex(path) for path in files}}


def _split_reservations(source, root):
    value = json.loads(_source_path(source['path'], root).read_text())
    if value.get('schema') != 'natlang.source-split-reservations/1':
        raise CustodyError('split reservations have an unsupported schema; custody fails closed')
    ids = set()
    for record in value['records']:
        ids.add(('id', record['id']))
        ids.update(('id', item) for item in record.get('source_ids') or [])
    return ids, {}


LOADERS = {'evaluation-packets': _evaluation_packets, 'source-groups': _source_groups,
           'decision-holds': _decision_holds, 'split-reservations': _split_reservations}

_PROTECTED_WORLDS_CACHE_LOCK = threading.RLock()
_PROTECTED_WORLDS_CACHE = {}
_PROTECTED_REGISTRY_CACHE = {}


def _custody_file_identity(path):
    """Five-field identity including a symlink's lexical and resolved targets."""
    lexical = Path(path).absolute()
    resolved = lexical.resolve(strict=True)
    link_stat = lexical.lstat()
    target_stat = resolved.stat()
    link_target = os.readlink(lexical) if stat.S_ISLNK(link_stat.st_mode) else None
    return (str(lexical), str(resolved), link_target,
            link_stat.st_dev, link_stat.st_ino, link_stat.st_size,
            link_stat.st_mtime_ns, link_stat.st_ctime_ns,
            target_stat.st_dev, target_stat.st_ino, target_stat.st_size,
            target_stat.st_mtime_ns, target_stat.st_ctime_ns,
            stat.S_IMODE(target_stat.st_mode))


def _protected_source_paths(declared, root):
    """Enumerate files whose contents feed ProtectedWorlds, without reading their rows."""
    paths = set()
    for source in declared['sources']:
        effective = dict(source)
        if 'root' in effective:
            from .paths import resolve
            effective['path'] = str(resolve(effective['root'], effective['path']))
        kind = effective.get('kind')
        if kind == 'evaluation-packets':
            registry_path = _source_path(effective['registry'], root)
            paths.add(registry_path)
            packet_registry = _read_protected_registry(registry_path)
            packet_entries = packet_registry.get('corpora', packet_registry) if isinstance(packet_registry, dict) else packet_registry
            for entry in packet_entries:
                if (isinstance(entry, dict) and entry.get('admission') == PROTECTED_ADMISSION
                        and 'cases.jsonl' in (entry.get('include') or [])):
                    paths.add(_source_path(entry['path'], root) / 'cases.jsonl')
        elif kind == 'decision-holds':
            directory = _source_path(effective['path'], root)
            paths.add(directory)
            paths.update(directory.glob('*.jsonl'))
        elif kind in {'source-groups', 'split-reservations'}:
            paths.add(_source_path(effective['path'], root))
        else:
            raise CustodyError(f'unknown protected source kind {kind!r}; custody fails closed')
    return tuple(sorted((_custody_file_identity(path) for path in paths), key=lambda item: item[0]))


def _protected_worlds_snapshot(registry_path, declared, root):
    return (_custody_file_identity(registry_path), *_protected_source_paths(declared, root))


def _read_protected_registry(path):
    key = str(Path(path).absolute())
    before = _custody_file_identity(path)
    with _PROTECTED_WORLDS_CACHE_LOCK:
        cached = _PROTECTED_REGISTRY_CACHE.get(key)
    if cached is not None and cached[0] == before:
        if before != _custody_file_identity(path):
            raise CustodyError(f'protected source registry {path} changed during cached verification')
        return cached[1]
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as error:
        raise CustodyError(f'protected source registry {path} is not readable; custody fails closed') from error
    if before != _custody_file_identity(path):
        raise CustodyError(f'protected source registry {path} changed while being read')
    with _PROTECTED_WORLDS_CACHE_LOCK:
        _PROTECTED_REGISTRY_CACHE[key] = (before, value)
    return value


@dataclass(frozen=True)
class ProtectedWorlds:
    """Exact protected identities of every declared protected source."""
    identities: frozenset
    sources: tuple  # canonical JSON receipts of the loaded sources

    @functools.cached_property
    def digest(self):  # immutable: hashing ~150k identities on every check() cost 0.2 s per case
        return canonical_json_sha256_hex({'sources': list(self.sources),
                                          'identities': sorted(map(list, self.identities))})

    @classmethod
    def from_cases(cls, packets):
        """``packets``: {packet_id: (cases_sha256, rows)} (fixtures and explicit packet sets)."""
        identities, bound = set(), []
        for packet_id in sorted(packets):
            digest, rows = packets[packet_id]
            for row in rows:
                identities |= world_identities(row, evidence_items=False)
            bound.append({'id': packet_id, 'cases_sha256': digest, 'rows': len(rows)})
        return cls(frozenset(identities), tuple(json.dumps(item, sort_keys=True) for item in bound))

    @classmethod
    def load(cls, root=None, registry=None):
        """Every source of the protected-worlds registry; fail closed on any unreadable or changed source."""
        root = (Path(root) if root is not None else _repo()).resolve()
        path = _source_path(registry or REGISTRY, root)
        try:
            declared = _read_protected_registry(path)
        except CustodyError as error:
            raise CustodyError(f'protected-worlds registry {path} is not readable; custody fails closed') from error
        if declared.get('schema') != REGISTRY_SCHEMA or not declared.get('sources'):
            raise CustodyError('protected-worlds registry declares no sources; custody fails closed')
        cache_key = (str(root), str(path.resolve(strict=True)))
        snapshot = _protected_worlds_snapshot(path, declared, root)
        with _PROTECTED_WORLDS_CACHE_LOCK:
            cached = _PROTECTED_WORLDS_CACHE.get(cache_key)
        if cached is not None and cached[0] == snapshot:
            # Recheck after lookup so a source change racing the cheap scan does
            # not return a stale protection set.
            if _protected_worlds_snapshot(path, declared, root) == snapshot:
                return cached[1]
        identities, receipts = set(), []
        for source in declared['sources']:
            loader = LOADERS.get(source.get('kind'))
            if loader is None:
                raise CustodyError(f'unknown protected source kind {source.get("kind")!r}; custody fails closed')
            if 'root' in source:  # a logical data root (common.paths), never a machine path in the registry
                from .paths import resolve
                source = {**source, 'path': str(resolve(source['root'], source['path']))}
            if 'sha256' in source:
                target = _source_path(source['path'], root)
                if not target.is_file() or sha256_file_hex(target) != source['sha256']:
                    raise CustodyError(f'protected source {source["id"]} is missing or changed; custody fails closed')
            try:
                found, detail = loader(source, root)
            except CustodyError:
                raise
            except (OSError, ValueError, KeyError) as error:
                raise CustodyError(f'protected source {source.get("id")} is not readable; custody fails closed') \
                    from error
            identities |= found
            receipts.append(json.dumps({'id': source['id'], 'kind': source['kind'], 'identities': len(found),
                                        **detail}, sort_keys=True))
        result = cls(frozenset(identities), tuple(receipts))
        if _protected_worlds_snapshot(path, declared, root) != snapshot:
            raise CustodyError('protected source files changed while custody was being verified; custody fails closed')
        with _PROTECTED_WORLDS_CACHE_LOCK:
            _PROTECTED_WORLDS_CACHE[cache_key] = (snapshot, result)
        return result

    def related(self, case):
        """Sorted identity kinds this case shares with a protected world (empty: unrelated)."""
        return sorted({kind for kind, _ in world_identities(case, evidence_items=False) & self.identities})

    def check(self, case):
        """Return the custody receipt of a training world, or raise CustodyError."""
        if not isinstance(case, dict) or not isinstance(case.get('id'), str):
            raise CustodyError('custody needs an identified case')
        if case.get('split') != 'train':
            raise CustodyError(f'custody refuses split {case.get("split")!r}: only train worlds teach or select')
        kinds = self.related(case)
        if kinds:
            raise CustodyError(f'custody refuses a world related to a protected world ({", ".join(kinds)})')
        return {'schema': CUSTODY_SCHEMA, 'case_id': case['id'], 'split': 'train',
                'world_sha256': world_fingerprint(case), 'protected_sha256': self.digest}


# ---------------------------------------------------------------- related worlds and slices

def related_components(rows, identities=world_identities):
    """Indices of rows grouped into related-world components, ordered by their smallest row id."""
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    first = {}
    for index, row in enumerate(rows):
        for identity in sorted(identities(row)):
            if identity in first:
                parents[find(index)] = find(first[identity])
            else:
                first[identity] = index
    groups = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return sorted(groups.values(), key=lambda indices: min(str(rows[index].get('id')) for index in indices))


def _slice_binding(rows):
    return {'rows': len(rows), 'ids_sha256': canonical_json_sha256_hex(sorted(row['id'] for row in rows)),
            'rows_sha256': canonical_json_sha256_hex([canonical_json_sha256_hex(row) for row in rows])}


def _disjoint(slices):
    seen = {}
    for name, rows in slices.items():
        if any(not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'] for row in rows):
            raise CustodyError(f'slice {name} has a row without an exact id')
        if len({row['id'] for row in rows}) != len(rows):
            raise CustodyError(f'slice {name} repeats an invocation id')
        identities = set().union(*(world_identities(row) for row in rows)) if rows else set()
        for other, known in seen.items():
            overlap = identities & known
            if overlap:
                kinds = ', '.join(sorted({kind for kind, _ in overlap}))
                raise CustodyError(f'slices {other} and {name} share related worlds ({kinds})')
        seen[name] = identities


def verify_slices(slices, *, protected):
    """Receipt for named slices that are pairwise unrelated and unrelated to every protected world."""
    if protected is None:
        raise CustodyError('slices need protected-world custody')
    _disjoint(slices)
    for rows in slices.values():
        for row in rows:
            protected.check(row)
    return {'schema': SLICES_SCHEMA, 'protected_sha256': protected.digest,
            'slices': {name: _slice_binding(rows) for name, rows in slices.items()}}


def check_slice_receipt(receipt, slices):
    """Re-verify a bound slice receipt where the protected sources are absent; returns the receipt."""
    if (not isinstance(receipt, dict) or receipt.get('schema') != SLICES_SCHEMA
            or not isinstance(receipt.get('protected_sha256'), str)
            or set(receipt.get('slices') or {}) != set(slices)
            or any(receipt['slices'][name] != _slice_binding(rows) for name, rows in slices.items())):
        raise CustodyError('slices differ from their protected-world custody receipt')
    _disjoint(slices)
    return receipt
