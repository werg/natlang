#!/usr/bin/env python3
"""Stage exact privately exported inputs for recovery; never grant admission.

Policy selects the catalog and destination. This mechanism authenticates the
original custody receipts, streams bounded downloads and inventories sealed TAR
members without extracting them or rewriting their original source identities.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import contextlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

try:
    from .private_export_guard import validate_receipt
except ImportError:
    from private_export_guard import validate_receipt


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def durable_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def catalog_jobs(catalog):
    jobs = {}
    for item in catalog['objects']:
        if item['role'] != 'bulk':
            continue
        receipt_path = Path(item['receipt'])
        receipt = json.loads(receipt_path.read_text())
        validate_receipt(receipt, item['identity'])
        if any(receipt[key] != item[key] for key in ('run', 'name')):
            raise ValueError('catalog object differs from original private receipt')
        identity = item['identity']
        job = jobs.setdefault(identity['sha256'], {
            'identity': identity, 'run': item['run'], 'name': item['name'],
            'sources': [], 'custody_receipts': []})
        if job['identity'] != identity:
            raise ValueError('conflicting content-addressed sizes')
        job['sources'].append(receipt['source'])
        job['custody_receipts'].append({'path': str(receipt_path), 'sha256': digest(receipt_path)})
    return list(jobs.values())


def write_verified(chunks, target, identity, *, reserve_bytes, resume=False, keep_partial=False):
    """Publish only complete matching bytes, leaving an existing file untouched."""
    target = Path(target)
    if target.is_symlink():
        raise ValueError('recovery target must not be a symlink')
    if target.exists():
        if target.stat().st_size != identity['bytes'] or digest(target) != identity['sha256']:
            raise ValueError('existing recovery object differs; refusing replacement')
        return True
    pending = target.with_name(target.name + '.pending')
    h, size = hashlib.sha256(), 0
    if pending.is_symlink():
        raise ValueError('recovery partial must not be a symlink')
    try:
        with pending.open('r+b' if resume and pending.exists() else 'xb') as stream:
            if resume:
                before = os.fstat(stream.fileno())
                for block in iter(lambda: stream.read(8 << 20), b''):
                    h.update(block)
                    size += len(block)
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('recovery partial changed during verification')
                if size > identity['bytes']:
                    raise ValueError('recovery partial exceeds declared size')
            for chunk in chunks:
                if not chunk:
                    continue
                if shutil.disk_usage(target.parent).free < reserve_bytes + len(chunk):
                    raise OSError('recovery staging reached its free-space reserve')
                size += len(chunk)
                if size > identity['bytes']:
                    raise ValueError('download exceeds declared object size')
                stream.write(chunk)
                h.update(chunk)
            if size != identity['bytes'] or h.hexdigest() != identity['sha256']:
                raise ValueError('download size or SHA-256 differs')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, target)
        fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException as error:
        # Never remove another invocation's pre-existing pending object.
        if 'stream' in locals() and not (keep_partial and not isinstance(error, ValueError)):
            pending.unlink(missing_ok=True)
        raise
    return False


def inventory_archive(path, identity):
    """Hash every regular member; preserve, but never follow, archive links."""
    import zstandard
    members, seen, source_manifest = [], set(), None
    with Path(path).open('rb') as source:
        magic = source.read(4)
        source.seek(0)
        if magic != b'\x28\xb5\x2f\xfd' and not magic.startswith(b'\x1f\x8b'):
            # The private input catalog also contains original JSON custody
            # receipts and roundtrip proofs. These are useful exact files,
            # not malformed archives; retain them without reinterpretation.
            return {'schema': 'natlang.recovery-archive-inventory/1', 'archive': identity,
                    'format': 'opaque-file', 'members': [], 'extracted': False,
                    'training_admission': False}
        compressed = magic == b'\x28\xb5\x2f\xfd'
        reader = zstandard.ZstdDecompressor().stream_reader(source) if compressed else source
        with reader if compressed else contextlib.nullcontext(reader) as decoded:
            with tarfile.open(fileobj=decoded, mode='r|' if compressed else 'r|gz') as archive:
                for member in archive:
                    name = PurePosixPath(member.name)
                    if name.is_absolute() or '..' in name.parts or member.name in seen:
                        raise ValueError('unsafe or duplicate recovery archive member')
                    seen.add(member.name)
                    if member.isdir():
                        continue
                    record = {'path': member.name, 'bytes': member.size}
                    if member.isfile():
                        stream = archive.extractfile(member)
                        if len(name.parts) == 2 and name.name == 'transfer-manifest.json':
                            if member.size > 64 * 2**20:
                                raise ValueError('recovery source manifest exceeds bounded metadata limit')
                            raw = stream.read()
                            record['sha256'] = hashlib.sha256(raw).hexdigest()
                            source_manifest = json.loads(raw)
                        else:
                            record['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
                    elif member.issym() or member.islnk():
                        record.update(link_target=member.linkname,
                                      kind='symlink' if member.issym() else 'hardlink')
                    else:
                        raise ValueError('unsupported recovery archive member kind')
                    members.append(record)
    if source_manifest is not None:
        if source_manifest.get('schema') != 'natlang.mined-source-transfer/1':
            raise ValueError('unsupported embedded source manifest')
        prefix = source_manifest['name'] + '/'
        expected = {}
        for item in source_manifest['files']:
            relative = PurePosixPath(item['path'])
            if relative.is_absolute() or '..' in relative.parts or item['path'] in expected:
                raise ValueError('unsafe or duplicate embedded source binding')
            expected[item['path']] = item
        actual = {m['path'][len(prefix):]: m for m in members
                  if m['path'].startswith(prefix) and m['path'] != prefix + 'transfer-manifest.json'}
        if set(actual) != set(expected) or any(
                any(actual[name].get(key) != item.get(key) for key in ('bytes', 'sha256'))
                for name, item in expected.items()):
            raise ValueError('embedded source manifest differs from actual member bytes')
    return {'schema': 'natlang.recovery-archive-inventory/1', 'archive': identity,
            'format': 'tar+zstd' if compressed else 'tar+gzip',
            'members': members, 'source_manifest': source_manifest,
            'extracted': False, 'training_admission': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--reserve-gib', type=int, default=60)
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error('reserve must be nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(args)


def _run(args):
    # Reuse the configured private account. Neither credentials nor signed
    # object URLs are written into manifests, journals or diagnostics.
    try:
        from . import cwctl
    except ImportError:
        import cwctl
    cwctl.load_env_file()
    import requests
    import wandb
    jobs = catalog_jobs(json.loads(args.catalog.read_text()))
    objects = args.out / 'objects'
    objects.mkdir(exist_ok=True)
    durable_json(args.out / 'plan.json', {
        'catalog': {'path': str(args.catalog), 'sha256': digest(args.catalog)},
        'jobs': jobs, 'training_admission': False, 'new_training_exposure': False})
    api, files = wandb.Api(timeout=60), {}
    for run in sorted({job['run'] for job in jobs}):
        names = [job['name'] for job in jobs if job['run'] == run]
        for offset in range(0, len(names), 100):
            for item in api.run(run).files(names=names[offset:offset + 100], per_page=200):
                files[(run, item.name)] = item

    def stage(job):
        identity = job['identity']
        target = objects / (identity['sha256'] + '.tar.zst')
        result = {'identity': identity, 'path': str(target), 'verified': False,
                  'inventory_verified': False, 'at': time.time()}
        try:
            if target.exists():
                reused = write_verified((), target, identity, reserve_bytes=0)
            else:
                file = files[(job['run'], job['name'])]
                if file.size != identity['bytes']:
                    raise ValueError('remote object metadata size differs')
                pending = target.with_name(target.name + '.pending')
                if pending.is_symlink():
                    raise ValueError('recovery partial must not be a symlink')
                offset = pending.stat().st_size if pending.exists() else 0
                if offset == identity['bytes']:
                    reused = write_verified((), target, identity, reserve_bytes=0, resume=True)
                else:
                    headers = {'Range': f'bytes={offset}-'} if offset else {}
                    with requests.get(file.url, auth=('api', os.environ['WANDB_API_KEY']), headers=headers,
                                      stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        if offset and (response.status_code != 206 or response.headers.get('Content-Range')
                                       != f"bytes {offset}-{identity['bytes'] - 1}/{identity['bytes']}"):
                            raise ValueError('resumed response lacks the exact requested byte range')
                        reused = write_verified(response.iter_content(1 << 20), target, identity,
                                                reserve_bytes=args.reserve_gib * 2**30,
                                                resume=bool(offset), keep_partial=True)
            result.update(verified=True, reused=reused)
            inventory = inventory_archive(target, identity)
            inventory_path = objects / (identity['sha256'] + '.inventory.json')
            durable_json(inventory_path, inventory)
            result.update(inventory_verified=True, inventory={
                'path': str(inventory_path), 'sha256': digest(inventory_path)},
                member_count=len(inventory['members']))
        except Exception as error:
            result['error_type'] = type(error).__name__
            # Exception text may contain signed URLs; retain no such text.
        return result

    results = []
    with (args.out / 'results.jsonl').open('a') as journal:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(stage, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                journal.write(json.dumps(result) + '\n')
                journal.flush()
                os.fsync(journal.fileno())
                summary = {'at': time.time(), 'total': len(jobs), 'finished': len(results),
                           'verified': sum(r['verified'] for r in results),
                           'inventoried': sum(r['inventory_verified'] for r in results),
                           'verified_bytes': sum(r['identity']['bytes'] for r in results if r['verified']),
                           'complete': len(results) == len(jobs), 'training_admission': False}
                durable_json(args.out / 'progress.json', summary)
                if len(results) % 20 == 0 or summary['complete']:
                    print(json.dumps(summary), flush=True)
    durable_json(args.out / 'result.json', {**summary, 'objects': results})
    if summary['inventoried'] != summary['total']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
