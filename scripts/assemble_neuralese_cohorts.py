#!/usr/bin/env python3
"""Assemble named training cohorts into one labelled trainer input, with a receipt.

Two modes, one contract (agreed DGX/Pop 2026-10-10):

- ``text``: text rows (natlang.native_gold_chat renderings, one document per row) concatenated with ``text_cohort``
  set to the cohort name; every row of every cohort must carry the same ``tokenizer_sha256``. Consumer:
  train.text_warmup ``--text-data`` with ``--cohort-weights``.
- ``records``: native records concatenated with ``cohort`` set to the cohort name, and the cohorts' pieces merged
  (one name, one text). Consumer: train.trajectories ``--records``/``--pieces`` with ``--cohort-weights``.

A cohort is ``NAME=corpus:<registered id>`` (training/neuralese_corpora.json, its immutable manifest under
training/corpus-manifests/ pins every byte; the entry must have ``training_admission: true``) or, for inputs that are
not registered corpora, ``NAME=path:<file>@<sha256>`` (records mode: ``NAME=path:<records>@<sha>,<pieces>@<sha>``)
together with ``--authority NAME=<text>``, the explicit lineage authority (recipe and review pins) under which that
input is already admitted. Every input byte is hashed while it is read. Record and row ids must be unique across
cohorts, a row's existing cohort label must match, and each cohort's held split must be source-disjoint from its
training split (source_groups). Assembly grants no admission: the receipt records sources, hashes and counts only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / 'training' / 'neuralese_corpora.json'
MANIFESTS = REPO / 'training' / 'corpus-manifests'
LABEL = {'text': 'text_cohort', 'records': 'cohort'}
COHORT_NAME_CHARS = set('abcdefghijklmnopqrstuvwxyz0123456789_-.')


def registered_file(corpus_id, name):
    """(path, sha256) of one file of an admitted registered corpus."""
    registry = json.loads(REGISTRY.read_text())
    entries = [entry for entry in registry['corpora'] if entry.get('id') == corpus_id]
    if len(entries) != 1:
        raise ValueError('not exactly one registered corpus: ' + corpus_id)
    entry = entries[0]
    if entry.get('training_admission') is not True:
        raise ValueError('registered corpus is not admitted for training: ' + corpus_id)
    manifest = json.loads((MANIFESTS / (corpus_id + '.json')).read_text())
    if manifest.get('id') != corpus_id:
        raise ValueError('manifest identity differs: ' + corpus_id)
    files = [item for item in manifest['files'] if item['path'] == name]
    if len(files) != 1:
        raise ValueError(f'{corpus_id} has no manifest file {name}')
    return REPO / entry['path'] / name, files[0]['sha256'], entry


def pinned(spec):
    path, sep, sha = spec.rpartition('@')
    if not sep or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha) or not path:
        raise ValueError('a path input is <file>@<lowercase sha256>: ' + spec)
    return Path(path), sha


def cohort_inputs(mode, name, source, authority):
    """{'role': (path, sha256)} and the source description for one cohort."""
    roles = ('text',) if mode == 'text' else ('records', 'pieces')
    kind, sep, value = source.partition(':')
    if kind == 'corpus' and sep:
        files = {'text': 'text.jsonl', 'records': 'records.jsonl', 'pieces': 'pieces.jsonl'}
        inputs, entry = {}, None
        for role in roles:
            path, sha, entry = registered_file(value, files[role])
            inputs[role] = (path, sha)
        return inputs, {'corpus': value, 'registry_admission': entry.get('admission')}
    if kind == 'path' and sep:
        if not authority:
            raise ValueError('path input of cohort ' + name + ' needs --authority ' + name + '=<lineage authority>')
        parts = value.split(',')
        if len(parts) != len(roles):
            raise ValueError(f'cohort {name}: {mode} mode takes {len(roles)} pinned path(s)')
        return {role: pinned(part) for role, part in zip(roles, parts)}, {'authority': authority}
    raise ValueError('cohort source is corpus:<id> or path:<file>@<sha256>: ' + source)


def hashed_lines(path, expected):
    """Yield the file's lines while hashing it; raise at the end when the bytes differ from the pin."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                yield line
    if digest.hexdigest() != expected:
        raise ValueError(f'{path}: sha256 {digest.hexdigest()} differs from the pinned {expected}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['text', 'records'])
    parser.add_argument('--cohort', action='append', required=True, metavar='NAME=SOURCE')
    parser.add_argument('--authority', action='append', default=[], metavar='NAME=TEXT')
    parser.add_argument('--limit', action='append', default=[], metavar='NAME=N',
                        help='keep only the first N rows of a cohort (smoke slices; recorded in the receipt)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)

    def named(values, what):
        result = {}
        for value in values:
            name, sep, rest = value.partition('=')
            if not sep or not name or not set(name) <= COHORT_NAME_CHARS or name in result:
                raise ValueError(f'invalid or duplicate {what}: {value}')
            result[name] = rest
        return result

    cohorts, authorities = named(args.cohort, '--cohort'), named(args.authority, '--authority')
    limits = {name: int(value) for name, value in named(args.limit, '--limit').items()}
    if set(authorities) - set(cohorts) or set(limits) - set(cohorts):
        raise ValueError('--authority/--limit name an undeclared cohort')
    if args.out.exists():
        raise ValueError('fresh output directory required')
    args.out.mkdir(parents=True)
    label = LABEL[args.mode]
    data_name = 'text.jsonl' if args.mode == 'text' else 'records.jsonl'
    receipt = {'schema': 'natlang.neuralese-cohort-assembly/1', 'mode': args.mode, 'label_field': label,
               'cohorts': {}, 'admission_granted': False,
               'note': 'Assembly records sources and counts only; admission is the recipe\'s and the registry\'s.'}
    ids, tokenizers, pieces = set(), set(), {}
    all_groups = {}
    data_path = args.out / data_name
    with data_path.open('w') as out:
        for name, source in cohorts.items():
            inputs, description = cohort_inputs(args.mode, name, source, authorities.get(name))
            rows, splits, groups = 0, {}, {}
            path, sha = inputs[args.mode if args.mode == 'text' else 'records']
            limit = limits.get(name)
            for line in hashed_lines(path, sha):
                if limit is not None and rows >= limit:
                    continue  # still hashed: the pin covers the whole file
                row = json.loads(line)
                if row.get(label, name) != name:
                    raise ValueError(f'{row.get("id")}: labelled {row[label]!r}, assembled as {name!r}')
                if not isinstance(row.get('id'), str) or row['id'] in ids:
                    raise ValueError('missing or duplicate id across cohorts: ' + str(row.get('id')))
                ids.add(row['id'])
                if args.mode == 'text':
                    if not isinstance(row.get('tokenizer_sha256'), str):
                        raise ValueError(row['id'] + ': text row without tokenizer_sha256')
                    tokenizers.add(row['tokenizer_sha256'])
                split = row.get('split')
                splits[split] = splits.get(split, 0) + 1
                for group in row.get('source_groups') or []:
                    groups.setdefault(group, set()).add(split)
                    all_groups.setdefault(group, set()).add(split)
                row[label] = name
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                rows += 1
            if args.mode == 'records':
                piece_path, piece_sha = inputs['pieces']
                for line in hashed_lines(piece_path, piece_sha):
                    piece = json.loads(line)
                    if pieces.setdefault(piece['name'], piece)['text'] != piece['text']:
                        raise ValueError('piece name with two texts across cohorts: ' + piece['name'])
            leaking = sorted(group for group, seen in groups.items() if len(seen) > 1)
            if leaking:
                raise ValueError(f'cohort {name}: held and training rows share source groups, e.g. {leaking[:3]}')
            if not splits.get('train'):
                raise ValueError(f'cohort {name} has no training rows')
            receipt['cohorts'][name] = {**description, 'rows': rows, 'splits': splits, 'limit': limit,
                                        'inputs': {role: {'path': str(path), 'sha256': sha}
                                                   for role, (path, sha) in inputs.items()}}
    leaking = sorted(group for group, seen in all_groups.items() if len(seen) > 1)
    if leaking:
        raise ValueError(f'assembled cohorts share source groups across splits, e.g. {leaking[:3]}')
    receipt['source_group_closure'] = {'groups': len(all_groups), 'cross_cohort_split_conflicts': 0,
                                       'scope': 'assembled inputs only; external corpus closure remains required'}
    if args.mode == 'text' and len(tokenizers) != 1:
        raise ValueError('cohorts were rendered with different tokenizers: ' + ', '.join(sorted(tokenizers)))
    if args.mode == 'text':
        receipt['tokenizer_sha256'] = tokenizers.pop()
    outputs = [data_path]
    if args.mode == 'records':
        piece_out = args.out / 'pieces.jsonl'
        piece_out.write_text(''.join(json.dumps(piece, ensure_ascii=False) + '\n' for piece in pieces.values()))
        outputs.append(piece_out)
    def file_sha256(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 24), b''):
                digest.update(block)
        return digest.hexdigest()
    receipt['outputs'] = {path.name: file_sha256(path) for path in outputs}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'out': str(args.out), 'outputs': receipt['outputs'],
                      'cohorts': {n: c['splits'] for n, c in receipt['cohorts'].items()}}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
