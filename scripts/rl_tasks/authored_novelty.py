#!/usr/bin/env python3
"""Repetition gate for an authored RL task batch against the existing task pool.

  authored_novelty.py BATCH_CASES.jsonl --pool POOL.jsonl [--pool ...] -o novelty.json

BATCH_CASES is the converted ``cases.jsonl`` that qualify_authored.py writes. A task is its program together with its
invocation inputs: several invocations of one program on materially different worlds are wanted (multi-operation
codebases call one program many times). Programs and inputs are compared after normalizing digits and quoted
literals (scripts/rl_tasks/measure_repetition.py), so a task whose program and inputs differ from another only in
names, numbers or quoted values is the same task.

- exact duplicate: same normalized program AND same normalized inputs as a pool task or an earlier batch task; a directory
  reducer's folder world counts exactly (raw file names and contents, hashed), so folder-only variants are not duplicates;
- near duplicate: within the same family, word 5-shingle Jaccard >= 0.8 between the normalized programs AND >= 0.8
  between the normalized inputs.

Text alone is not a program: renaming the root or reformatting a sentence changes the text, not the program. Programs
are therefore also counted as functional programs (measure_repetition.FunctionalPrograms): the same structure (call
graph, helper argument/return types, kind/readout, helper calls, spelled-out code steps and reference scripts,
literals abstracted) AND word-trigram Jaccard >= 0.7 of the normalized prose. The threshold is calibrated on the audited
round-1 batches against the auditors' distinct-program counts (10 of 12 batches within 30%); it is a proxy, and a
rule sentence toggled inside one helper can be undercounted. A batch program whose text is new to its family but which
is functionally an existing program is reported as ``functional_repeats``.

The batch PASSES when it has no exact duplicates, at most 10% near duplicates, and every family in the batch,
together with the same family already in the pool, has >= 20 distinct normalized programs, or the batch adds >= 5
distinct programs to it; and, in each family, at least half of the batch's new programs (at least one) are
functionally new, unless the family already has >= 20 functional signatures. Distinct programs and functional
signatures per family are reported separately from tasks.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_repetition import FunctionalPrograms, normalize, program_features, write_text_atomic  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve_str  # noqa: E402


def shingles(text, size=5):
    words = text.split()
    return {' '.join(words[i:i + size]) for i in range(max(1, len(words) - size + 1))}


def normalize_value(value):
    """Inputs with every string normalized like program text (digits, quoted literals inside the text)."""
    if isinstance(value, str):
        return normalize(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return '#'
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {normalize(str(key)): normalize_value(item) for key, item in value.items()}
    return value


def folder_digest(folder_files):
    """Exact identity of a directory reducer's world: raw file names (digits and all) and raw file contents."""
    canonical = json.dumps(sorted((str(path), str(text)) for path, text in folder_files.items()), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def task_key(row, features):
    """Exact key: normalized program, normalized inputs, and for a directory reducer the exact folder world (file names
    and contents, not digit-normalized: ``01.md``/``02.md`` entries or a long and a short world of one program are
    different invocations, where the old key collapsed them into exact duplicates). Digits and quoted literals in the
    plain inputs still do not make a different task."""
    semantics = row.get('semantics') or {}
    inputs = normalize_value(semantics.get('inputs'))
    folder = semantics.get('folder_files')
    if folder:
        inputs = {'inputs': inputs, 'folder_sha256': folder_digest(folder)}
    return features['program'], json.dumps(inputs, sort_keys=True, ensure_ascii=False)


def near_key(row, features):
    """Near-duplicate text: the exact key's program and inputs, with the folder world as normalized file names and text,
    so two worlds that differ only in a few notes still compare by their shared wording."""
    semantics = row.get('semantics') or {}
    inputs = {'inputs': normalize_value(semantics.get('inputs'))}
    if semantics.get('folder_files'):
        inputs['folder_files'] = {str(path): normalize(str(text)) for path, text in sorted(semantics['folder_files'].items())}
    return features['program'], json.dumps(inputs, sort_keys=True, ensure_ascii=False)


def rows(path):
    with open(path) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


STATIC_POOL_BYTES = 50_000_000  # pools at least this large are indexed once and reused (the admitted pool changes each batch)
INDEX_ROOT = Path(os.environ.get('NATLANG_NOVELTY_INDEX') or resolve_str('data_hdd', 'natlang-rl-authoring', 'novelty-index'))


def crc_shingles(text):
    return frozenset(zlib.crc32(item.encode()) for item in shingles(text))


def digest(value):
    return hashlib.sha1(value.encode()).digest()


def build_index(path):
    """Everything the gate needs from one pool file, in a form that is small to load: exact-key digests, per-family
    program digests, the functional adds to replay, and per-family near-duplicate shingle hashes (CRC32 of the 5-word
    shingles: the Jaccard test is the same, a 32-bit collision between shingles of one family is negligible)."""
    tasks, programs, functional, near = set(), collections.defaultdict(set), [], collections.defaultdict(list)
    for row in rows(path):
        features = program_features(row)
        key = task_key(row, features)
        tasks.add(digest(key[0] + '\0' + key[1]))
        programs[features['family']].add(digest(features['program']))
        functional.append((features['family'], features['functional'], features['functional_words']))
        text = near_key(row, features)
        near[features['family']].append((crc_shingles(text[0]), crc_shingles(text[1])))
    return {'tasks': tasks, 'programs': dict(programs), 'functional': functional, 'near': dict(near)}


def pool_index(path, families):
    """The index of ``path`` restricted to ``families`` (the batch's): everything in the gate is keyed by family, so a
    large static pool is cached once under INDEX_ROOT (keyed by path, size and mtime) as a small shared part plus one shard
    per family, and a run loads only the shards of the families it checks."""
    stat = Path(path).stat()
    if stat.st_size < STATIC_POOL_BYTES:
        index = build_index(path)
        return {'tasks': index['tasks'], 'programs': index['programs'],
                'functional': _group(index['functional']), 'near': index['near']}
    import pickle
    name = hashlib.sha1(f'{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}'.encode()).hexdigest()
    root = INDEX_ROOT / name
    if not (root / 'meta.pkl').exists():
        index = build_index(path)
        grouped = _group(index['functional'])
        root.mkdir(parents=True, exist_ok=True)
        for family in set(grouped) | set(index['near']):
            with open(root / f'{digest(family).hex()}.pkl', 'wb') as handle:
                pickle.dump({'functional': grouped.get(family, []), 'near': index['near'].get(family, [])}, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        pending = root / 'meta.pending'
        with open(pending, 'wb') as handle:  # last: a reader never sees a half-built index
            pickle.dump({'tasks': index['tasks'], 'programs': index['programs']}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pending.replace(root / 'meta.pkl')
    with open(root / 'meta.pkl', 'rb') as handle:
        meta = pickle.load(handle)
    out = {'tasks': meta['tasks'], 'programs': meta['programs'], 'functional': {}, 'near': {}}
    for family in families:
        shard = root / f'{digest(family).hex()}.pkl'
        if shard.exists():
            with open(shard, 'rb') as handle:
                data = pickle.load(handle)
            out['functional'][family], out['near'][family] = data['functional'], data['near']
    return out


def _group(functional):
    grouped = collections.defaultdict(list)
    for family, structure, words in functional:
        grouped[family].append((structure, words))
    return dict(grouped)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('batch')
    parser.add_argument('--pool', action='append', default=[])
    parser.add_argument('-o', '--out')
    args = parser.parse_args(argv)
    batch = [(row, program_features(row)) for row in rows(args.batch)]
    families_checked = {features['family'] for _, features in batch}
    pool_tasks, pool_family_programs = set(), collections.defaultdict(set)
    functional = FunctionalPrograms()
    pool_shingles = collections.defaultdict(list)
    for path in args.pool:
        index = pool_index(path, families_checked)
        pool_tasks |= index['tasks']
        for family, items in index['programs'].items():
            pool_family_programs[family] |= items
        for family, items in index['functional'].items():
            for structure, words in items:
                functional.add(family, structure, words)
        for family, items in index['near'].items():
            pool_shingles[family].extend(items)
    pool_functional = collections.Counter(functional.counts)
    exact, near, seen = [], [], set()
    added = collections.defaultdict(set)
    new_functional = collections.Counter()
    functional_repeats, seen_programs = [], collections.defaultdict(set)
    for row, features in batch:
        family = features['family']
        is_new = functional.add(family, features['functional'], features['functional_words'])
        new_functional[family] += is_new
        program_id = digest(features['program'])
        if not is_new and program_id not in pool_family_programs[family] | seen_programs[family]:
            functional_repeats.append(row['id'])
        seen_programs[family].add(program_id)
    for row, features in batch:
        program = digest(features['program'])
        key = task_key(row, features)
        key = digest(key[0] + '\0' + key[1])
        if key in pool_tasks or key in seen:
            exact.append(row['id'])
        seen.add(key)
        keyed = near_key(row, features)
        mine = (crc_shingles(keyed[0]), crc_shingles(keyed[1]))
        for other in pool_shingles[features['family']]:
            if all(len(a & b) / max(1, len(a | b)) >= 0.8 for a, b in zip(mine, other)):
                near.append(row['id'])
                break
        pool_shingles[features['family']].append(mine)
        added[features['family']].add(program)
    families = {}
    for family, programs in added.items():
        total = len(pool_family_programs[family] | programs)
        new_programs = len(programs - pool_family_programs[family])
        batch_functional = new_functional[family]
        functional_total = pool_functional[family] + batch_functional
        functional_ok = (functional_total >= 20 or new_programs == 0
                         or (batch_functional >= 1 and 2 * batch_functional >= new_programs))
        families[family] = {'pool_programs': len(pool_family_programs[family]),
                            'batch_tasks': sum(1 for _, f in batch if f['family'] == family),
                            'batch_new_programs': new_programs, 'total': total,
                            'pool_functional': pool_functional[family],
                            'batch_new_functional': batch_functional, 'functional_total': functional_total,
                            'text_ok': total >= 20 or new_programs >= 5, 'functional_ok': functional_ok}
        families[family]['ok'] = families[family]['text_ok'] and functional_ok
    near_share = len(near) / len(batch) if batch else 0.
    result = {'schema': 'natlang.rl-authored-novelty/1', 'batch': str(Path(args.batch).resolve()),
              'pool': [str(Path(p).resolve()) for p in args.pool], 'tasks': len(batch),
              'exact_duplicates': exact, 'near_duplicates': near, 'near_duplicate_share': round(near_share, 4),
              'functional_repeats': functional_repeats,
              'families': families}
    result['result'] = ('PASS' if batch and not exact and near_share <= 0.1 and all(f['ok'] for f in families.values())
                        else 'FAIL')
    text = json.dumps(result, indent=1)
    if args.out:
        write_text_atomic(args.out, text + '\n')
        print(f'wrote {Path(args.out).resolve()}', file=sys.stderr)
    print(text)
    return 0 if result['result'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
