#!/usr/bin/env python3
"""Acquire a bounded, attributable pilot. Never execute downloaded source code.

Pinned task downloads and content-addressed dataset-server trajectory snapshots are
kept in an ignored cache. A snapshot is NOT represented as a pinned parquet export.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile
import tarfile

WORKBENCH = '49c7dfd00c03d384ec59ea57374f50b766aa5613'
TATQA = '870accc41953dcde885aabeb963d94aabdc0fbc3'
COMMITPACK = 'fc56fe33c030c6daa414c2b112c932b8eed085e6'
TREEDST = '7816c25a9613a8877924e5cdc4f6ce565774fdd9'
SCIFACT = '68b98a56d93e0f9da0d2aab4e6c3294699a0f72e'
QASPER = 'fdc9d8214fbab5dd782958601db4d678e6934a54'
PERMISSIVE = {'mit', 'apache-2.0', 'bsd-2-clause', 'bsd-3-clause', 'isc', '0bsd'}


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'natlang-source-pilot/1'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def acquire(cache, sources, limit, trajectory_rows):
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'version': 'natlang.directory_sources/1', 'sources': {}}

    def save(key, rows, evidence, license):
        raw = (json.dumps(rows, ensure_ascii=False) + '\n').encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = f'{key}-{digest}.json'
        (cache / path).write_bytes(raw)
        return key, {'path': path, 'sha256': digest, 'rows': len(rows), 'license': license, **evidence}

    def task_source(key):
        if key in {'qasper', 'scifact', 'treedst'}:
            url, revision, terms = {
                'qasper': ('https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz', QASPER, 'CC-BY-4.0'),
                'scifact': ('https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz', SCIFACT, 'CC-BY-4.0 annotations; ODC-By-1.0 abstracts'),
                'treedst': (f'https://raw.githubusercontent.com/apple/ml-tree-dst/{TREEDST}/dataset/treedst.zip', TREEDST, 'CC-BY-SA-3.0'),
            }[key]
            # Never extract or execute archives. Preserve the original bytes, including held-out partitions.
            raw = fetch(url)
            archive_digest = hashlib.sha256(raw).hexdigest()
            (cache / f'{key}-{archive_digest}.archive').write_bytes(raw)
            if key == 'treedst':
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    members = {p: archive.read(p) for p in archive.namelist() if not p.endswith('/')}
            else:
                with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
                    members = {p.name: archive.extractfile(p).read() for p in archive.getmembers() if p.isfile()}
            def member(suffix):
                return next(value for name, value in members.items() if name.endswith(suffix))
            held = []
            if key == 'qasper':
                train = json.loads(member('qasper-train-v0.3.json'))
                held = list(json.loads(member('qasper-dev-v0.3.json')))
                rows = [{'id': k, **v} for k, v in train.items()][:limit * 3]
            elif key == 'scifact':
                parse = lambda suffix: [json.loads(line) for line in member(suffix).splitlines() if line.strip()]
                corpus = parse('corpus.jsonl')
                dev = parse('claims_dev.jsonl')
                held = sorted({str(doc) for row in dev for doc in row.get('cited_doc_ids', [])} |
                              {str(doc) for row in dev for doc in row.get('evidence', {})})
                rows = [{'claims': parse('claims_train.jsonl')[:limit * 3], 'corpus': corpus}]
            else:
                # Source formats are inspected by the strict adapter; retain schema alongside original train rows.
                train_names = [p for p in members if 'train' in p.lower() and p.endswith(('.json', '.jsonl'))]
                if len(train_names) != 1:
                    raise ValueError(f'ambiguous_treedst_train_members:{train_names}')
                blob = members[train_names[0]].decode()
                # The author's *_dst.json files contain one JSON object per line.
                parsed = [json.loads(line) for line in blob.splitlines() if line.strip()]
                rows = parsed[:limit * 3] if isinstance(parsed, list) else [{'id': k, **v} for k, v in parsed.items()][:limit * 3]
            return save(key, rows, {'revision': revision, 'original_split': 'train', 'held_out_ids': held,
                'capture': 'content-addressed-author-release',
                'files': [{'url': url, 'sha256': archive_digest, 'members': sorted(members)}]}, terms)
        if key == 'workbench':
            base = f'https://raw.githubusercontent.com/olly-styles/WorkBench/{WORKBENCH}/'
            paths = ['data/processed/emails.csv', 'data/processed/tasks_and_outcomes/email_tasks_and_outcomes.csv']
            blobs = [fetch(base + path) for path in paths]
            emails, tasks = [list(csv.DictReader(io.StringIO(b.decode()))) for b in blobs]
            return save(key, [{'emails': emails, 'tasks': tasks}], {
                'revision': WORKBENCH, 'original_split': 'unsplit',
                'files': [{'url': base + p, 'sha256': hashlib.sha256(b).hexdigest()} for p, b in zip(paths, blobs)]}, 'MIT')
        if key == 'tatqa':
            url = f'https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{TATQA}/dataset_raw/tatqa_dataset_train.json'
            raw = fetch(url)
            return save(key, json.loads(raw)[:limit * 3], {'revision': TATQA, 'original_split': 'train',
                'files': [{'url': url, 'sha256': hashlib.sha256(raw).hexdigest()}]}, 'CC-BY-4.0')
        if key == 'commitpack':
            rows, files = [], []
            for language in ['markdown', 'yaml', 'json']:
                url = f'https://huggingface.co/datasets/bigcode/commitpackft/resolve/{COMMITPACK}/data/{language}/data.jsonl'
                scanned = 0
                with urllib.request.urlopen(url, timeout=120) as response:
                    for line in response:
                        scanned += 1
                        row = json.loads(line)
                        license = str(row.get('license', '')).lower()
                        if license in PERMISSIVE and row.get('old_contents') and row.get('new_contents'):
                            rows.append({**row, 'language': language})
                        if sum(r['language'] == language for r in rows) >= limit or scanned >= 3000:
                            break
                files.append({'url': url, 'sampled_lines': scanned, 'capture': 'bounded-prefix'})
            return save(key, rows, {'revision': COMMITPACK, 'original_split': 'train', 'files': files}, 'MIT; per-row repository license')
        if key == 'musique':
            url = 'https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t'
            archive_path = cache / 'musique-v1.0.zip'
            if not archive_path.exists() or not zipfile.is_zipfile(archive_path):
                staged = archive_path.with_name(archive_path.name + f'.partial-{os.getpid()}')
                try:
                    with urllib.request.urlopen(url, timeout=120) as response, staged.open('wb') as output:
                        for chunk in iter(lambda: response.read(1024 * 1024), b''):
                            output.write(chunk)
                    if not zipfile.is_zipfile(staged):
                        raise ValueError('MuSiQue download is not a complete archive')
                    staged.replace(archive_path)
                finally:
                    staged.unlink(missing_ok=True)
            with archive_path.open('rb') as stream:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(chunk)
                digest = hasher.hexdigest()
            with zipfile.ZipFile(archive_path) as archive:
                path = next(p for p in archive.namelist() if p.endswith('musique_full_v1.0_train.jsonl'))
                rows = []
                with archive.open(path) as stream:
                    for line in stream:
                        rows.append(json.loads(line))
                        if len(rows) >= limit * 3:
                            break
                held_seeds = set()
                dev_path = next(p for p in archive.namelist() if p.endswith('musique_full_v1.0_dev.jsonl'))
                with archive.open(dev_path) as stream:
                    for line in stream:
                        for question in json.loads(line).get('question_decomposition', []):
                            held_seeds.add(str(question['id']))
            return save(key, rows, {'revision': 'musique-v1.0', 'original_split': 'train',
                'held_out_seed_ids': sorted(held_seeds),
                'files': [{'url': url, 'sha256': digest, 'member': path}]}, 'CC-BY-4.0')
        raise ValueError(f'unknown task source {key}')

    def traces(key):
        dataset, config, split = {
            'nebius': ('nebius/SWE-rebench-openhands-trajectories', 'default', 'train'),
            'swesmith': ('SWE-bench/SWE-smith-trajectories', 'default', 'tool'),
            'nvidia': ('nvidia/Open-SWE-Traces', 'v1.1', 'openhands'),
        }[key]
        info_url = 'https://huggingface.co/api/datasets/' + dataset
        revision = json.loads(fetch(info_url))['sha']
        rows, files = [], []
        for offset in range(0, trajectory_rows, 20):
            url = 'https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode({
                'dataset': dataset, 'config': config, 'split': split,
                'offset': offset, 'length': min(20, trajectory_rows - offset)})
            raw = fetch(url)
            result = json.loads(raw)
            for entry in result['rows']:
                if entry.get('truncated_cells'):
                    rows.append({'capture_error': 'truncated_cells', 'row_idx': entry['row_idx']})
                else:
                    rows.append({**entry['row'], '_source_row_index': entry['row_idx']})
            files.append({'url': url, 'sha256': hashlib.sha256(raw).hexdigest()})
        if json.loads(fetch(info_url))['sha'] != revision:
            raise ValueError(f'{dataset} changed during acquisition')
        # Preserve original repository terms for candidate slices; never infer them from the trace license.
        licenses = {}
        for row in rows:
            if row.get('resolved') not in [True, 1]:
                continue
            repo = row.get('repo')
            if not repo and key == 'swesmith':
                repo = str(row.get('instance_id', '')).split('.')[0].replace('__', '/')
            if not repo or repo in licenses:
                continue
            try:
                api = json.loads(fetch(f'https://api.github.com/repos/{repo}/license'))
                licenses[repo] = {'spdx': api['license']['spdx_id'], 'url': api['html_url'],
                    'license_blob_sha': api['sha'], 'scope': 'current-repository-license; historical terms need review'}
            except Exception as error:
                licenses[repo] = {'error': str(error)}
        return save(key, rows, {'dataset': dataset, 'metadata_revision': revision,
            'capture': 'content-addressed-dataset-server-snapshot', 'config': config, 'original_split': split,
            'files': files, 'repository_licenses': licenses}, 'MIT' if key == 'swesmith' else 'CC-BY-4.0')

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(traces if key in {'nebius', 'swesmith', 'nvidia'} else task_source, key): key for key in sources}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                key, record = future.result()
                manifest['sources'][key] = record
                manifest.get('failures', {}).pop(key, None)
                print(key, record['rows'], record['sha256'][:12], flush=True)
            except Exception as error:
                manifest.setdefault('failures', {})[key] = str(error)
                print(key, 'FAILED', error, flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('vendor/directory-sources'))
    parser.add_argument('--sources', default='workbench,tatqa,commitpack,musique,nebius,swesmith,nvidia')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--trajectory-rows', type=int, default=60)
    args = parser.parse_args()
    if args.limit < 1 or not 1 <= args.trajectory_rows <= 500:
        parser.error('limit must be positive; trajectory-rows must be 1..500')
    acquire(args.cache, args.sources.split(','), args.limit, args.trajectory_rows)
