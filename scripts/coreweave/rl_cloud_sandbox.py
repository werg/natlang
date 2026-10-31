#!/usr/bin/env python3
"""Sandbox side of the cloud RL lineages (scripts/coreweave/rl_cloud.py; plans/neuralese/LFM3_RL_CLOUD_20261031.md).

Stdlib only (runs before and beside the job's own environments). No credential ever reaches this process: uploads and
downloads use per-object signed URLs (capabilities) that the DGX issues with the storage account's key.

  stage LINEAGE_DIR EXPORTS_DIR       after a completed round: an immutable export of the lineage boundary
  upload CAPABILITY.json              PUT the listed staged blobs to their signed URLs (events as JSON lines)
  fetch PARENT.json LINEAGE_DIR       materialize a parent export from signed GET URLs, verifying every byte
  prune LINEAGE_DIR EXPORTS_DIR       drop staged exports the DGX has uploaded and superseded, and superseded checkpoints

An export is the lineage state at a round boundary as content-addressed blobs:
  top        tar.zst of the lineage's top-level files and actor-tokenizers/ (state, config, selections, bindings)
  round      tar.zst of one completed round directory (rollouts, receipts, learner metadata), without checkpoints
             and without the regenerable actor export (actor-store/); made once per round and carried forward
  checkpoint the parent full-state checkpoint (weights, optimizer, RNG, exposure), raw
Blobs are hard links (raw) or written files under EXPORTS_DIR/cas/<sha256>, so a later round cannot change what a
staged export names. A manifest EXPORTS_DIR/<seq>.json lists them; the DGX exporter uploads the newest one.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

SCHEMA = 'natlang.rl-cloud-export/1'
ROUND = re.compile(r'round-(\d{6})(-attempt-\d+)?$')
SKIP = ('.pending', '.partial', '.tmp', '.reserve')


# The shared stdlib-only helpers of the code snapshot (natlang_neuralese.common), usable before any pip install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as file_digest  # noqa: E402
from natlang_neuralese.common.jsonio import write_json_durable  # noqa: E402


def event(**fields):
    print(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields}), flush=True)


def _tar_zst(base, members, out):
    """Deterministic member order; zstd multi-threaded. Returns (sha256, bytes) of the compressed blob."""
    listing = tempfile.NamedTemporaryFile('w', delete=False, dir=out.parent)
    listing.write(''.join(f'{m}\n' for m in sorted(members)))
    listing.close()
    try:
        tar = subprocess.Popen(['tar', '-C', str(base), '--no-recursion', '-cf', '-', '-T', listing.name],
                               stdout=subprocess.PIPE)
        with open(out, 'wb') as sink:
            zstd = subprocess.run(['zstd', '-q', '-3', '-T0', '-c'], stdin=tar.stdout, stdout=sink)
        tar.stdout.close()
        if tar.wait() != 0 or zstd.returncode != 0:
            raise RuntimeError(f'archive of {base} failed')
    finally:
        os.unlink(listing.name)
    digest = file_digest(out)
    return digest, out.stat().st_size


def _members(root, top_only=False, exclude=()):
    out = []
    root = Path(root)
    for path in sorted(root.rglob('*') if not top_only else root.iterdir()):
        rel = path.relative_to(root).as_posix()
        if any(rel == e or rel.startswith(e + '/') for e in exclude) or rel.endswith(SKIP):
            continue
        if path.is_dir() and not path.is_symlink():
            out.append(rel)
        elif path.is_file():
            out.append(rel)
    return out


def _cas_put_archive(cas, base, members):
    temp = cas / f'.archive-{os.getpid()}-{time.time_ns()}'
    digest, size = _tar_zst(base, members, temp)
    target = cas / digest
    if target.exists():
        temp.unlink()
    else:
        temp.replace(target)
    return digest, size


def stage(lineage, exports):
    lineage, exports = Path(lineage).resolve(), Path(exports)
    cas = exports / 'cas'
    cas.mkdir(parents=True, exist_ok=True)
    has_state = (lineage / 'state.json').is_file()
    # A lineage before its first round (the CPU pipeline check): the seeded parent is the boundary.
    state = (json.loads((lineage / 'state.json').read_text()) if has_state
             else {'checkpoint': str(lineage / 'parent' / 'checkpoint.pt'), 'round': 0, 'trained_rounds': 0})
    previous = sorted(exports.glob('[0-9]*.json'))
    prior = json.loads(previous[-1].read_text()) if previous else {}
    if prior.get('trained_rounds') == state.get('trained_rounds') and prior.get('round') == state.get('round'):
        event(event='stage_skipped', reason='boundary already staged', seq=prior['seq'])
        return prior
    blobs = []
    # top level: files plus actor-tokenizers/
    top = [p.name for p in lineage.iterdir() if p.is_file() and p.name != 'controller.lock' and not p.name.endswith(SKIP)]
    if (lineage / 'actor-tokenizers').is_dir():
        top += ['actor-tokenizers'] + [f'actor-tokenizers/{m}' for m in _members(lineage / 'actor-tokenizers')]
    digest, size = _cas_put_archive(cas, lineage, top)
    blobs.append({'kind': 'top', 'sha256': digest, 'bytes': size, 'encoding': 'tar.zst'})
    # completed rounds, once each (carried forward from the previous export)
    archived = {b['round']: b for b in prior.get('blobs', []) if b['kind'] == 'round'}
    current = int(state.get('round', 0))
    for path in sorted(p for p in lineage.iterdir() if p.is_dir() and ROUND.match(p.name)):
        if int(ROUND.match(path.name).group(1)) >= current:
            continue
        if path.name in archived and (cas / archived[path.name]['sha256']).exists():
            blobs.append(archived[path.name])
            continue
        if path.name in archived:  # uploaded and pruned locally: still named by this export
            blobs.append(archived[path.name])
            continue
        members = [path.name] + [f'{path.name}/{m}' for m in _members(path, exclude=('actor-store',))
                                 if not (m.startswith('learner/') and (m.endswith('.pt') or m == 'learner/rl-native-objective.json'))]
        digest, size = _cas_put_archive(cas, lineage, members)
        blobs.append({'kind': 'round', 'round': path.name, 'sha256': digest, 'bytes': size, 'encoding': 'tar.zst'})
    # parent checkpoint (raw, hard link) and its aliases in the same learner directory
    checkpoint = Path(state['checkpoint'])
    links = []
    if checkpoint.is_file():
        stat = checkpoint.stat()
        digest = file_digest(checkpoint)
        target = cas / digest
        if not target.exists():
            os.link(checkpoint, target)
        blobs.append({'kind': 'checkpoint', 'sha256': digest, 'bytes': stat.st_size, 'encoding': 'raw',
                      'path': str(checkpoint)})
        for alias in sorted(checkpoint.parent.glob('*.pt')):
            if alias != checkpoint:
                a = alias.stat()
                same = (a.st_dev, a.st_ino) == (stat.st_dev, stat.st_ino) or file_digest(alias) == digest
                if same:
                    links.append({'path': str(alias), 'same_as': str(checkpoint)})
    # Sequence numbers continue the lineage's exports across sandboxes (fetch records the parent's).
    base = exports / 'parent-seq'
    parent_seq = int(base.read_text()) if base.exists() else -1
    seq = max(prior.get('seq', -1), parent_seq) + 1
    manifest = {'schema': SCHEMA, 'seq': seq, 'lineage': lineage.name, 'lineage_dir': str(lineage),
                'created': time.time(), 'round': current, 'trained_rounds': state.get('trained_rounds', 0),
                'phase': state.get('phase'), 'state_sha256': file_digest(lineage / 'state.json') if has_state else None,
                'checkpoint': str(checkpoint), 'blobs': blobs, 'links': links,
                'previous_seq': prior.get('seq')}
    write_json_durable(exports / f'{seq:06d}.json', manifest)
    event(event='staged', seq=seq, trained_rounds=manifest['trained_rounds'], blobs=len(blobs),
          bytes=sum(b['bytes'] for b in blobs))
    return manifest


class _Body:
    """File body with a fixed length for urllib (streams; never reads the file into memory)."""
    def __init__(self, path):
        self.stream = open(path, 'rb')

    def read(self, size=-1):
        return self.stream.read(size if size and size > 0 else 8 << 20)


def upload(capability):
    cap = json.loads(Path(capability).read_text())
    failed = 0
    for item in cap['items']:
        path = Path(item['source'])
        if path.stat().st_size != item['bytes']:
            raise ValueError(f'staged blob size changed: {path}')
        for attempt in range(4):
            started = time.time()
            body = _Body(path)
            request = urllib.request.Request(item['url'], data=body, method='PUT',
                                             headers={**item.get('headers', {}), 'Content-Length': str(item['bytes'])})
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    status = response.status
            except Exception as error:  # retried; the DGX re-verifies every byte on its GET
                status = getattr(error, 'code', None) or f'{type(error).__name__}'
            finally:
                body.stream.close()
            seconds = round(time.time() - started, 1)
            event(event='uploaded' if status in (200, 201, 204) else 'upload_error', sha256=item['sha256'],
                  bytes=item['bytes'], status=status, seconds=seconds, attempt=attempt,
                  mb_per_s=round(item['bytes'] / 1e6 / max(seconds, 0.1), 1))
            if status in (200, 201, 204):
                break
            time.sleep(10 * (attempt + 1))
        else:
            failed += 1
    return 1 if failed else 0


def _download(url, destination, sha256, size):
    for attempt in range(5):
        digest, written = hashlib.sha256(), 0
        started = time.time()
        try:
            with urllib.request.urlopen(url, timeout=900) as response, open(destination, 'wb') as out:
                for chunk in iter(lambda: response.read(8 << 20), b''):
                    digest.update(chunk)
                    out.write(chunk)
                    written += len(chunk)
            if written == size and digest.hexdigest() == sha256:
                seconds = round(time.time() - started, 1)
                event(event='fetched', sha256=sha256, bytes=size, seconds=seconds,
                      mb_per_s=round(size / 1e6 / max(seconds, 0.1), 1))
                return
            event(event='fetch_mismatch', sha256=sha256, written=written, attempt=attempt)
        except Exception as error:
            event(event='fetch_error', sha256=sha256, error=f'{type(error).__name__}: {error}'[:300], attempt=attempt)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'could not fetch blob {sha256}')


def fetch(parent, lineage):
    parent = json.loads(Path(parent).read_text())
    lineage = Path(lineage)
    if parent.get('schema') != 'natlang.rl-cloud-parent/1':
        raise ValueError('not a cloud RL parent capability')
    if lineage.resolve() != Path(parent['lineage_dir']).resolve():
        raise ValueError('parent export belongs to another lineage directory')
    lineage.mkdir(parents=True, exist_ok=True)
    scratch = lineage.parent / f'.fetch-{lineage.name}'
    scratch.mkdir(exist_ok=True)
    for blob in parent['blobs']:
        temp = scratch / blob['sha256']
        if not (temp.exists() and temp.stat().st_size == blob['bytes'] and file_digest(temp) == blob['sha256']):
            _download(blob['url'], temp, blob['sha256'], blob['bytes'])
        if blob['encoding'] == 'raw':
            target = Path(blob['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp, target)
        else:
            tar = subprocess.run(f'zstd -q -dc {temp} | tar -x -C {lineage}', shell=True)
            if tar.returncode != 0:
                raise RuntimeError(f'cannot unpack {blob["kind"]} blob {blob["sha256"]}')
            temp.unlink()
    for link in parent.get('links', []):
        target = Path(link['path'])
        if target.exists():
            target.unlink()
        os.link(link['same_as'], target)
    state = lineage / 'state.json'
    if parent.get('state_sha256') and file_digest(state) != parent['state_sha256']:
        raise ValueError('restored state.json differs from the export')
    shutil.rmtree(scratch, ignore_errors=True)
    exports = lineage.parent / '.exports' / lineage.name
    exports.mkdir(parents=True, exist_ok=True)
    (exports / 'parent-seq').write_text(str(parent['seq'] if isinstance(parent.get('seq'), int) else -1))
    event(event='parent_restored', seq=parent.get('seq'), trained_rounds=parent.get('trained_rounds'))
    return 0


def prune(lineage, exports):
    """Keep the newest uploaded export's blobs and anything newer; drop older staged exports and superseded checkpoints
    (prune_rl.sh semantics: checkpoints of rounds other than the current parent's)."""
    lineage, exports = Path(lineage).resolve(), Path(exports)
    uploaded = sorted(int(p.stem.split('.')[0]) for p in exports.glob('*.uploaded'))
    if not uploaded:
        return 0
    newest = uploaded[-1]
    keep = set()
    for path in exports.glob('[0-9]*.json'):
        manifest = json.loads(path.read_text())
        if manifest['seq'] >= newest:
            keep |= {b['sha256'] for b in manifest['blobs']}
    removed = 0
    for blob in (exports / 'cas').iterdir():
        if blob.name not in keep and not blob.name.startswith('.'):
            removed += blob.stat().st_size
            blob.unlink()
    state = json.loads((lineage / 'state.json').read_text())
    parent = Path(state['checkpoint']).resolve().parent.parent
    staged = json.loads((exports / f'{newest:06d}.json').read_text())
    if Path(staged['checkpoint']).resolve().parent.parent == parent:  # the current parent is safely exported
        for path in lineage.iterdir():
            match = ROUND.match(path.name)
            if not match or path == parent or int(match.group(1)) >= int(state['round']):
                continue
            for name in ('learner/checkpoint.pt', 'learner/best-checkpoint.pt', 'learner/rl-native-objective.json'):
                target = path / name
                if target.exists():
                    removed += target.stat().st_size
                    target.unlink()
            shutil.rmtree(path / 'actor-store' / 'sglang-hf', ignore_errors=True)
    event(event='pruned', newest_uploaded=newest, bytes=removed)
    return 0


def main(argv):
    command, *args = argv
    if command == 'stage':
        stage(*args)
        return 0
    if command == 'upload':
        return upload(*args)
    if command == 'fetch':
        return fetch(*args)
    if command == 'prune':
        return prune(*args)
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
