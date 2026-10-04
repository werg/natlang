#!/usr/bin/env python3
"""Acquire bounded, revision-pinned visual task sources without executing dataset code.

Uses the stored HF login when available. Never prints tokens. Binary/source files
stay in the development dataset mirror; task preparation is a separate step.
"""
import argparse
import hashlib
import json
import shutil
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            result.remove_header('Authorization')
        return result


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    staged.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    staged.replace(path)


def safe_path(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '\\' in value or any(part in ('.', '..') for part in value.split('/')):
        raise ValueError('unsafe upstream path')
    return path


def contained(root, path):
    resolved_root = root.resolve()
    if not path.resolve().is_relative_to(resolved_root):
        raise ValueError('path escapes declared artifact root')
    return path


def acquire(source, root, opener, token):
    if len(source['revision']) != 40 or any(c not in '0123456789abcdef' for c in source['revision']):
        raise ValueError('source needs an immutable commit SHA')
    safe_path(source['id'])
    if '/' in source['id']:
        raise ValueError('source ID must be one path component')
    directory = contained(root, root / source['id'] / source['revision'])
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = contained(directory, directory / 'acquisition.json')
    old = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    identity = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if old and old['source_spec_sha256'] != identity:
        raise ValueError('source lock changed; use a new acquisition directory')
    receipt = old or {'schema': 'natlang.visual-source-acquisition/1', 'source_spec_sha256': identity,
        'source': source, 'files': [], 'status': 'in_progress', 'executed_upstream_code': False}
    known = {row['path']: row for row in receipt['files']}
    files = list(source['files'])
    if source.get('dataset_card_sha256'):
        files.insert(0, 'README.md')
    write_json(receipt_path, receipt)
    try:
        for name in files:
            path = contained(directory, directory / 'files' / safe_path(name))
            if name in known:
                if not path.is_file() or digest(path) != known[name]['sha256']:
                    raise ValueError('committed acquisition bytes changed: ' + name)
                continue
            if shutil.disk_usage(directory).free < source['max_file_bytes'] + 2_000_000_000:
                raise ValueError('insufficient disk reserve for bounded source download')
            url = source['upstream_url'] + '/resolve/' + source['revision'] + '/' + urllib.parse.quote(name, safe='/')
            headers = {'User-Agent': 'natlang-visual-source-acquisition/1'}
            if token:
                headers['Authorization'] = 'Bearer ' + token
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + '.partial')
            for attempt in range(4):
                try:
                    with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as response, partial.open('wb') as stream:
                        declared_size = int(response.headers.get('Content-Length', 0))
                        if declared_size > source['max_file_bytes']:
                            raise ValueError('upstream file exceeds per-file byte cap: ' + name)
                        size = 0
                        for block in iter(lambda: response.read(1024 * 1024), b''):
                            size += len(block)
                            if size > source['max_file_bytes']:
                                raise ValueError('upstream stream exceeds per-file byte cap: ' + name)
                            stream.write(block)
                        if declared_size and size != declared_size:
                            raise ConnectionError('upstream response ended before declared size')
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            sha = digest(partial)
            if name == 'README.md' and sha != source['dataset_card_sha256']:
                raise ValueError('dataset card differs from pinned review bytes')
            partial.replace(path)
            item = {'path': name, 'sha256': sha, 'bytes': size}
            receipt['files'].append(item)
            known[name] = item
            write_json(receipt_path, receipt)
            print(json.dumps({'source': source['id'], 'file': name, 'bytes': size}), flush=True)
        receipt.update(status='complete', errors=[])
    except Exception as error:
        receipt.update(status='incomplete', errors=[{'type': type(error).__name__, 'message': str(error)}])
        write_json(receipt_path, receipt)
        raise
    write_json(receipt_path, receipt)
    return {'id': source['id'], 'receipt': str(receipt_path), 'receipt_sha256': digest(receipt_path),
        'files': len(receipt['files']), 'bytes': sum(row['bytes'] for row in receipt['files']), 'status': receipt['status']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=Path('training/visual_sources.json'))
    parser.add_argument('--out', type=Path, default=Path('vendor/datasets/visual-frontend'))
    parser.add_argument('--source', default='all', help='Comma-separated source IDs or all')
    parser.add_argument('--index-only', action='store_true', help='Rebuild the mirror index from durable receipts without downloading')
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    selected = set(args.source.split(','))
    sources = [row for row in registry['sources'] if args.source == 'all' or row['id'] in selected]
    if args.source != 'all' and selected != {row['id'] for row in sources}:
        parser.error('unknown source ID')
    try:
        from huggingface_hub import get_token
        token = get_token()
    except ImportError:
        token = None
    opener = urllib.request.build_opener(SafeRedirect())
    results = []
    for source in ([] if args.index_only else sources):
        try:
            results.append(acquire(source, args.out, opener, token))
        except Exception as error:
            results.append({'id': source['id'], 'status': 'incomplete', 'error': str(error)})
            print(json.dumps(results[-1]), flush=True)
    observed = []
    for source in registry['sources']:
        receipt_path = contained(args.out, args.out / safe_path(source['id']) / source['revision'] / 'acquisition.json')
        item = {'id': source['id'], 'status': 'not_acquired'}
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            item.update(status=receipt['status'], receipt=str(receipt_path), receipt_sha256=digest(receipt_path), files=len(receipt['files']), bytes=sum(f['bytes'] for f in receipt['files']))
        observed.append(item)
    write_json(args.out / 'acquisition-index.json', {'schema': 'natlang.visual-source-index/1',
        'registry_sha256': digest(args.registry), 'sources': observed, 'unavailable': registry.get('unavailable', [])})
    raise SystemExit(1 if any(row['status'] != 'complete' for row in results) else 0)


if __name__ == '__main__':
    main()
