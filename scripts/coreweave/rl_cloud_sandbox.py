#!/usr/bin/env python3
"""Sandbox side of the cloud lineages (scripts/coreweave/rl_cloud.py; plans/neuralese/LFM3_RL_CLOUD_20261031.md).

Stdlib only (runs before and beside the job's own environments). No credential ever reaches this process: uploads and
downloads use per-object signed URLs (capabilities) that the DGX issues with the storage account's key.

  stage LINEAGE_DIR EXPORTS_DIR       an immutable export of the lineage's current boundary (skipped when unchanged)
  upload CAPABILITY.json              PUT the listed staged blobs to their signed URLs (events as JSON lines)
  fetch PARENT.json LINEAGE_DIR       materialize a parent export and the run inputs from signed GET URLs, verifying
                                      every byte (exact ZIP-delta checkpoints are reconstructed over their full bases)
  prune LINEAGE_DIR EXPORTS_DIR       drop staged exports the DGX has uploaded and superseded, and superseded checkpoints
  activate LINEAGE_DIR OUT CONFIG     make an SFT window (or its new control) the active controller
  progress LINEAGE_DIR                the active controller and whether its last call earned durable progress (JSON)

A lineage is either one RL controller at LINEAGE_DIR (``config.json``; progress = trained rounds) or a joint-SFT
lineage of windows (``lineage.json``; each window a mandatory supervised controller, progress = a completed window).
An export is the boundary as content-addressed blobs:
  top        tar.zst of the lineage's top-level files, actor-tokenizers/ and an RL controller's metadata
  round      tar.zst of one completed RL round directory (rollouts, receipts, learner metadata), without checkpoints
             and without the regenerable actor export (actor-store/); made once per round and carried forward
  window     tar.zst of one SFT window's metadata (state, controls, learner reports/receipts, resident plan) without
             checkpoints; a completed window's once (carried forward), the active window's at every boundary
  checkpoint full-state checkpoints (weights, optimizer, RNG, exposure), raw: the resume point (a completed window's
             or RL update's pending checkpoint, an interrupted window's rolling slot, else the controller parent) and
             what its resume binds (an interrupted window's exact parent)
Blobs are hard links (raw) or written files under EXPORTS_DIR/cas/<sha256>, so a later boundary cannot change what a
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
# Blobs travel as parts (private objects blobs/<sha256>.p0000 ...): a failed transfer repeats one part, not a 15 GB
# checkpoint (the DGX uplink is a few MB/s and drops long uploads).
PART_BYTES = 1 << 30


def part_name(digest, index):
    return f'blobs/{digest}.p{index:04d}'


def digest_parts(path, part_bytes=PART_BYTES):
    """(whole sha256, [{'index', 'offset', 'bytes', 'sha256'}]) in one read of the file."""
    whole, parts = hashlib.sha256(), []
    with open(path, 'rb') as stream:
        offset = 0
        while True:
            digest, size = hashlib.sha256(), 0
            while size < part_bytes:
                chunk = stream.read(min(8 << 20, part_bytes - size))
                if not chunk:
                    break
                digest.update(chunk)
                whole.update(chunk)
                size += len(chunk)
            if size == 0 and parts:
                break
            parts.append({'index': len(parts), 'offset': offset, 'bytes': size, 'sha256': digest.hexdigest()})
            offset += size
            if size < part_bytes:
                break
    return whole.hexdigest(), parts
ROUND = re.compile(r'round-(\d{6})(-attempt-\d+)?$')
SKIP = ('.pending', '.partial', '.tmp', '.reserve')


# The shared stdlib-only helpers of the code snapshot (natlang_neuralese.common), usable before any pip install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as file_digest  # noqa: E402
from natlang_neuralese.common.jsonio import write_json_durable  # noqa: E402


def event(**fields):
    print(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields}), flush=True)


def _tar_zst(base, members, out):
    """Deterministic member order; zstd multi-threaded (the CLI, else the Python zstandard module: the bootstrap's apt
    install of the CLI can fail or hang)."""
    listing = tempfile.NamedTemporaryFile('w', delete=False, dir=out.parent)
    listing.write(''.join(f'{m}\n' for m in sorted(members)))
    listing.close()
    try:
        tar = subprocess.Popen(['tar', '-C', str(base), '--no-recursion', '-cf', '-', '-T', listing.name],
                               stdout=subprocess.PIPE)
        with open(out, 'wb') as sink:
            if shutil.which('zstd'):
                ok = subprocess.run(['zstd', '-q', '-3', '-T0', '-c'], stdin=tar.stdout, stdout=sink).returncode == 0
            else:
                import zstandard
                zstandard.ZstdCompressor(level=3, threads=-1).copy_stream(tar.stdout, sink)
                ok = True
        tar.stdout.close()
        if tar.wait() != 0 or not ok:
            raise RuntimeError(f'archive of {base} failed')
    finally:
        os.unlink(listing.name)


def untar_zst(blob, directory):
    """Unpack a tar.zst blob into ``directory`` (zstd CLI, else the Python zstandard module)."""
    tar = subprocess.Popen(['tar', '-x', '-C', str(directory)], stdin=subprocess.PIPE)
    with open(blob, 'rb') as source:
        if shutil.which('zstd'):
            ok = subprocess.run(['zstd', '-q', '-dc'], stdin=source, stdout=tar.stdin).returncode == 0
        else:
            import zstandard
            zstandard.ZstdDecompressor().copy_stream(source, tar.stdin)
            ok = True
    tar.stdin.close()
    if tar.wait() != 0 or not ok:
        raise RuntimeError(f'cannot unpack {blob}')


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
    _tar_zst(base, members, temp)
    digest, parts = digest_parts(temp)
    size = temp.stat().st_size
    target = cas / digest
    if target.exists():
        temp.unlink()
    else:
        temp.replace(target)
    return digest, size, parts


LINEAGE_SCHEMA = 'natlang.cloud-sft-windows/1'


def controller(lineage):
    """(out, config) of the lineage's active controller. An RL lineage is one controller at the lineage directory
    (``config.json``); a joint-SFT lineage is a sequence of windows (``lineage.json``), the last one active."""
    lineage = Path(os.path.abspath(lineage))
    index = lineage / 'lineage.json'
    if index.is_file():
        value = json.loads(index.read_text())
        if value.get('schema') != LINEAGE_SCHEMA or not value.get('windows'):
            raise ValueError(f'{index} is not a {LINEAGE_SCHEMA} index')
        active = value['windows'][-1]
        return Path(active['out']), Path(active['config'])
    return lineage, lineage / 'config.json'


def activate(lineage, out, config):
    """Make (out, config) the active window: a new window is appended; the same window takes a new control (a bound
    rolling resume). Earlier windows stay listed, so the lineage keeps its whole history."""
    lineage = Path(os.path.abspath(lineage))
    index = lineage / 'lineage.json'
    value = json.loads(index.read_text())
    if value.get('schema') != LINEAGE_SCHEMA:
        raise ValueError(f'{index} is not a {LINEAGE_SCHEMA} index')
    out, config = os.path.abspath(out), os.path.abspath(config)
    if not Path(config).is_file() or json.loads(Path(config).read_text()).get('out') != out:
        raise ValueError(f'{config} is not a control of {out}')
    windows = value['windows']
    if windows and windows[-1]['out'] == out:
        windows[-1] = {**windows[-1], 'config': config, 'controls': windows[-1].get('controls', []) + [windows[-1]['config']]}
    elif any(w['out'] == out for w in windows):
        raise ValueError(f'{out} is an earlier window')
    else:
        windows.append({'index': len(windows), 'out': out, 'config': config})
    write_json_durable(index, value)
    event(event='window_active', index=len(windows) - 1, out=out, config=config)


def _state(out, config):
    path = out / 'state.json'
    if path.is_file():
        return json.loads(path.read_text()), True
    # before the controller's first call: its configured parent is the boundary
    return {'checkpoint': config['checkpoint'], 'round': 0, 'trained_rounds': 0}, False


def resume_checkpoints(out, config, state):
    """The checkpoints a resume of this boundary needs, the resume point first.

    A completed SFT window (or an RL update awaiting publication) resumes from ``pending_native_checkpoint``. An SFT
    window interrupted after rolling checkpoints resumes in its own slot through ``rolling_sft_resume``, which binds
    that slot and the window's exact parent, so both travel. Otherwise the controller's parent is the boundary."""
    pending = state.get('pending_native_checkpoint')
    if pending and Path(pending).is_file():
        return [Path(pending)]
    parent = Path(state['checkpoint'])
    slot = out / 'learner' / 'checkpoint.pt'
    if config.get('mandatory_sft_only') and slot.is_file() and not (parent.exists() and os.path.samefile(slot, parent)):
        return [slot, parent]
    # An interleaved RL controller's active SFT window (rl/interleave.py) resumes in that window's own rolling slot
    # (interleave._rolling_control); the slot and the window parent travel like a plain window's.
    active = (state.get('interleave') or {}).get('active_window')
    if config.get('interleaved_rl_stage') and active:
        slot = Path(active['directory']) / 'learner' / 'checkpoint.pt'
        if slot.is_file() and not (parent.exists() and os.path.samefile(slot, parent)):
            return [slot, parent]
    return [parent]


def progress(lineage):
    """The lineage's boundary as the driver sees it (JSON): which controller is active and whether its last call
    earned durable progress (a trained RL round or a completed SFT window with optimizer updates)."""
    out, config_path = controller(lineage)
    config = json.loads(config_path.read_text())
    state, _ = _state(out, config)
    window_updates = None
    report = out / 'learner' / 'report.json'
    if state.get('phase') == 'awaiting_semantic_refresh' and report.is_file():
        boundary = json.loads(report.read_text()).get('continuation_boundary') or {}
        if boundary.get('reached') is True:
            window_updates = boundary.get('new_optimizer_updates', 0)
    rolling = None
    if config.get('mandatory_sft_only') and not state.get('pending_native_checkpoint'):
        checkpoints = resume_checkpoints(out, config, state)
        rolling = len(checkpoints) == 2
    return {'out': str(out), 'config': str(config_path), 'kind': 'sft' if config.get('mandatory_sft_only') else 'rl',
            'phase': state.get('phase'), 'trained_rounds': state.get('trained_rounds', 0),
            'window_complete': window_updates is not None and window_updates > 0, 'window_updates': window_updates,
            'rolling_slot': rolling, 'bound_rolling_resume': bool(config.get('rolling_sft_resume'))}


def _stamp(path):
    stat = Path(path).stat()
    return {'path': str(path), 'bytes': stat.st_size, 'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns}


def _cas_put_link(cas, path):
    """A raw blob as a hard link of the file's current inode: the content a writer later replaces atomically (a new
    inode) is not what the export names. Hashed through the link, so a concurrent replacement cannot mix contents."""
    temp = cas / f'.link-{os.getpid()}-{time.time_ns()}'
    os.link(path, temp)
    try:
        before = temp.stat()
        digest, parts = digest_parts(temp)
        after = temp.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'{path} changed in place while it was hashed')
        target = cas / digest
        if target.exists():
            temp.unlink()
        else:
            temp.replace(target)
        return digest, after.st_size, parts
    finally:
        if temp.exists():
            temp.unlink()


def _metadata_members(base, directory):
    """A controller directory's metadata (state, control inputs, learner reports and receipts, resident plan) without
    checkpoints, reservations, locks and the regenerable actor export."""
    out = []
    for rel in _members(directory, exclude=('actor-store', 'learner/actor-view-owner')):
        name = rel.rsplit('/', 1)[-1]
        if (name.endswith('.pt') or name.startswith('.') or '.pt.reserv' in name or name.endswith('.lock')
                or rel.startswith('round-')):
            continue
        out.append(f'{directory.relative_to(base).as_posix()}/{rel}' if directory != base else rel)
    return out


TUNE_EXCLUDE = ('hash-cache.jsonl', 'natlang-config')


def tune_progress(stage_dir):
    """What a tune shard has actually produced: per-program evaluation arms, tune/timing/failure records, entry and log
    sizes (natlang_neuralese.tune.stage writes each step's result before the next starts)."""
    stage_dir = Path(stage_dir)
    programs = {}
    for program in sorted((stage_dir / 'programs').glob('*/')):
        evaluations = program / '.natlang' / 'tune' / 'evaluations'
        arms = {part.name: sorted(p.stem for p in part.glob('*.json')) for part in sorted(evaluations.glob('*/'))} \
            if evaluations.is_dir() else {}
        if arms:
            programs[program.name] = arms
    entry = stage_dir / 'entry.json'
    return {'programs': programs, 'tunes': sorted(p.name for p in (stage_dir / 'tunes').glob('*.json')),
            'shards': sorted(p.name for p in (stage_dir / 'shards').glob('*.json')),
            'entry': json.loads(entry.read_text()) if entry.is_file() else None,
            'logs': {p.name: p.stat().st_size for p in sorted((stage_dir / 'logs').glob('*'))}}


def stage_tune(lineage, exports):
    """An export of a tune shard: its command and status, and the stage directory (without its hash cache and tuner
    configuration home), staged when anything in it changed."""
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    cas = exports / 'cas'
    cas.mkdir(parents=True, exist_ok=True)
    spec = json.loads((lineage / 'tune.json').read_text())
    status_path = lineage / 'tune-status.json'
    status = json.loads(status_path.read_text()) if status_path.is_file() else {'status': 'running'}
    stage_dir = Path(spec['stage_dir'])
    members = [m for m in _members(stage_dir, exclude=TUNE_EXCLUDE)] if stage_dir.is_dir() else []
    listing = [(m, (stage_dir / m).stat().st_size, (stage_dir / m).stat().st_mtime_ns) for m in members
               if (stage_dir / m).is_file()]
    boundary = {'tree': canonical_json_sha256_hex(listing), 'status': status}
    previous = sorted(exports.glob('[0-9]*.json'))
    prior = json.loads(previous[-1].read_text()) if previous else {}
    if prior.get('boundary') == boundary:
        event(event='stage_skipped', reason='boundary already staged', seq=prior['seq'])
        return prior
    top = [p.name for p in lineage.iterdir() if p.is_file() and not p.name.endswith(SKIP)]
    digest, size, parts = _cas_put_archive(cas, lineage, top)
    blobs = [{'kind': 'top', 'sha256': digest, 'bytes': size, 'encoding': 'tar.zst', 'parts': parts}]
    if members:
        digest, size, parts = _cas_put_archive(cas, stage_dir, members)
        blobs.append({'kind': 'stage', 'path': str(stage_dir), 'sha256': digest, 'bytes': size, 'encoding': 'tar.zst',
                      'parts': parts})
    base = exports / 'parent-seq'
    seq = max(prior.get('seq', -1), int(base.read_text()) if base.exists() else -1) + 1
    manifest = {'schema': SCHEMA, 'seq': seq, 'lineage': lineage.name, 'lineage_dir': str(lineage), 'kind': 'tune',
                'created': time.time(), 'round': 0, 'trained_rounds': 0, 'phase': status['status'],
                'final': status['status'] in ('done', 'failed'), 'state_sha256': None, 'boundary': boundary,
                'checkpoint': None, 'checkpoints': [], 'tune_progress': tune_progress(stage_dir),
                'blobs': blobs, 'links': [], 'previous_seq': prior.get('seq')}
    write_json_durable(exports / f'{seq:06d}.json', manifest)
    event(event='staged', seq=seq, phase=manifest['phase'], blobs=len(blobs), bytes=sum(b['bytes'] for b in blobs),
          tunes=manifest['tune_progress']['tunes'])
    return manifest


def stage(lineage, exports):
    # The logical (DGX) path, never resolved: in a sandbox the DGX home root is a symlink onto the scratch volume, and the
    # export names the paths the DGX hub copy and a resumed sandbox use.
    lineage, exports = Path(os.path.abspath(lineage)), Path(exports)
    if (lineage / 'tune.json').is_file():
        return stage_tune(lineage, exports)
    cas = exports / 'cas'
    cas.mkdir(parents=True, exist_ok=True)
    out, config_path = controller(lineage)
    config = json.loads(config_path.read_text())
    state, has_state = _state(out, config)
    checkpoints = resume_checkpoints(out, config, state)
    boundary = {'controller': str(out), 'config_sha256': file_digest(config_path),
                'state_sha256': file_digest(out / 'state.json') if has_state else None,
                'checkpoints': [_stamp(p) for p in checkpoints if p.is_file()]}
    previous = sorted(exports.glob('[0-9]*.json'))
    prior = json.loads(previous[-1].read_text()) if previous else {}
    if prior.get('boundary') == boundary:
        event(event='stage_skipped', reason='boundary already staged', seq=prior['seq'])
        return prior
    blobs = []
    windows = json.loads((lineage / 'lineage.json').read_text())['windows'] if (lineage / 'lineage.json').is_file() else []
    # top level: files plus actor-tokenizers/
    top = [p.name for p in lineage.iterdir() if p.is_file() and p.name != 'controller.lock' and not p.name.endswith(SKIP)]
    if (lineage / 'actor-tokenizers').is_dir():
        top += ['actor-tokenizers'] + [f'actor-tokenizers/{m}' for m in _members(lineage / 'actor-tokenizers')]
    if not windows:  # an RL lineage's controller metadata lives at the top (and in its rounds)
        top += [m for m in _metadata_members(lineage, lineage) if '/' in m and not m.startswith('actor-tokenizers')]
    digest, size, parts = _cas_put_archive(cas, lineage, top)
    blobs.append({'kind': 'top', 'sha256': digest, 'bytes': size, 'encoding': 'tar.zst', 'parts': parts})
    # SFT windows: a completed window's metadata once (carried forward), the active window's at every boundary
    final = {b['window']: b for b in prior.get('blobs', []) if b['kind'] == 'window' and b.get('final')}
    for index, window in enumerate(windows):
        directory = Path(window['out'])
        complete = index < len(windows) - 1
        if complete and window['out'] in final:
            blobs.append(final[window['out']])
            continue
        if not directory.is_dir():
            continue
        members = [directory.relative_to(lineage).as_posix()] + _metadata_members(lineage, directory)
        digest, size, parts = _cas_put_archive(cas, lineage, members)
        blobs.append({'kind': 'window', 'window': window['out'], 'final': complete, 'sha256': digest, 'bytes': size,
                      'encoding': 'tar.zst', 'parts': parts})
    # RL rounds: completed rounds, once each (carried forward from the previous export)
    archived = {b['round']: b for b in prior.get('blobs', []) if b['kind'] == 'round'}
    current = int(state.get('round', 0))
    for path in sorted(p for p in out.iterdir() if p.is_dir() and ROUND.match(p.name)) if not windows else []:
        if int(ROUND.match(path.name).group(1)) >= current:
            continue
        if path.name in archived:  # staged before (perhaps uploaded and pruned locally): still named by this export
            blobs.append(archived[path.name])
            continue
        members = [path.name] + [f'{path.name}/{m}' for m in _members(path, exclude=('actor-store',))
                                 if not (m.startswith('learner/') and (m.endswith('.pt') or m == 'learner/rl-native-objective.json'))]
        digest, size, parts = _cas_put_archive(cas, lineage, members)
        blobs.append({'kind': 'round', 'round': path.name, 'sha256': digest, 'bytes': size, 'encoding': 'tar.zst',
                      'parts': parts})
    # checkpoints (raw, hard links): the resume point first, then what its resume control binds
    links = []
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            continue
        digest, size, parts = _cas_put_link(cas, checkpoint)
        blobs.append({'kind': 'checkpoint', 'sha256': digest, 'bytes': size, 'encoding': 'raw',
                      'path': str(checkpoint), 'parts': parts})
        stat = checkpoint.stat()
        for alias in sorted(checkpoint.parent.glob('*.pt')):
            if alias != checkpoint and alias not in checkpoints:
                a = alias.stat()
                if (a.st_dev, a.st_ino) == (stat.st_dev, stat.st_ino) or (a.st_size == size and file_digest(alias) == digest):
                    links.append({'path': str(alias), 'same_as': str(checkpoint)})
    # Sequence numbers continue the lineage's exports across sandboxes (fetch records the parent's).
    base = exports / 'parent-seq'
    parent_seq = int(base.read_text()) if base.exists() else -1
    seq = max(prior.get('seq', -1), parent_seq) + 1
    manifest = {'schema': SCHEMA, 'seq': seq, 'lineage': lineage.name, 'lineage_dir': str(lineage),
                'created': time.time(), 'round': current, 'trained_rounds': state.get('trained_rounds', 0),
                'phase': state.get('phase'), 'controller': str(out), 'window': len(windows) - 1 if windows else None,
                'state_sha256': boundary['state_sha256'], 'boundary': boundary,
                'checkpoint': str(checkpoints[0]), 'checkpoints': [str(p) for p in checkpoints],
                'blobs': blobs, 'links': links, 'previous_seq': prior.get('seq')}
    write_json_durable(exports / f'{seq:06d}.json', manifest)
    event(event='staged', seq=seq, phase=manifest['phase'], window=manifest['window'],
          trained_rounds=manifest['trained_rounds'], blobs=len(blobs), bytes=sum(b['bytes'] for b in blobs))
    return manifest


class _Body:
    """A byte range of a file as a fixed-length urllib body (streams; never reads it into memory)."""
    def __init__(self, path, offset, length):
        self.stream = open(path, 'rb')
        self.stream.seek(offset)
        self.left = length

    def read(self, size=-1):
        size = self.left if size is None or size < 0 else min(size, self.left)
        chunk = self.stream.read(min(size, 8 << 20))
        self.left -= len(chunk)
        return chunk


def put_part(url, headers, path, offset, length, log_fields, attempts=4):
    """PUT one part to its signed URL, with retries. True on success."""
    for attempt in range(attempts):
        started = time.time()
        body = _Body(path, offset, length)
        request = urllib.request.Request(url, data=body, method='PUT', headers={**headers, 'Content-Length': str(length)})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                status = response.status
        except Exception as error:  # retried; the DGX re-verifies every byte on its GET
            status = getattr(error, 'code', None) or f'{type(error).__name__}: {error}'[:200]
        finally:
            body.stream.close()
        seconds = round(time.time() - started, 1)
        event(event='uploaded' if status in (200, 201, 204) else 'upload_error', status=status, seconds=seconds,
              attempt=attempt, bytes=length, mb_per_s=round(length / 1e6 / max(seconds, 0.1), 1), **log_fields)
        if status in (200, 201, 204):
            return True
        time.sleep(10 * (attempt + 1))
    return False


def upload(capability):
    """Items: one per part {name, sha256 (part), source, offset, bytes, url, headers}."""
    cap = json.loads(Path(capability).read_text())
    failed = 0
    for item in cap['items']:
        path = Path(item['source'])
        if path.stat().st_size < item['offset'] + item['bytes']:
            raise ValueError(f'staged blob shorter than its parts: {path}')
        if not put_part(item['url'], item.get('headers', {}), path, item['offset'], item['bytes'],
                        {'name': item['name'], 'sha256': item['sha256']}):
            failed += 1
    return 1 if failed else 0


def download_parts(parts, destination, sha256, size):
    """GET every part (signed URL) into its offset of ``destination``, each part and the whole verified."""
    with open(destination, 'wb') as out:
        out.truncate(size)
    for part in parts:
        for attempt in range(5):
            digest, written = hashlib.sha256(), 0
            started = time.time()
            try:
                with urllib.request.urlopen(part['url'], timeout=900) as response, open(destination, 'r+b') as out:
                    out.seek(part['offset'])
                    for chunk in iter(lambda: response.read(8 << 20), b''):
                        digest.update(chunk)
                        out.write(chunk)
                        written += len(chunk)
                if written == part['bytes'] and digest.hexdigest() == part['sha256']:
                    seconds = round(time.time() - started, 1)
                    event(event='fetched', blob=sha256, part=part['index'], bytes=written, seconds=seconds,
                          mb_per_s=round(written / 1e6 / max(seconds, 0.1), 1))
                    break
                event(event='fetch_mismatch', blob=sha256, part=part['index'], written=written, attempt=attempt)
            except Exception as error:
                event(event='fetch_error', blob=sha256, part=part['index'],
                      error=f'{type(error).__name__}: {error}'[:300], attempt=attempt)
            time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f'could not fetch part {part["index"]} of blob {sha256}')
    if file_digest(destination) != sha256:
        raise RuntimeError(f'blob {sha256}: parts verified but the whole differs')


def _fetch_verified(entry, destination):
    """One blob (``sha256``, ``bytes``, ``parts`` with signed URLs) at ``destination``; reused when already exact."""
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size == entry['bytes'] and file_digest(destination) == entry['sha256']:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f'.{destination.name}.fetch')
    download_parts(entry['parts'], temp, entry['sha256'], entry['bytes'])
    os.replace(temp, destination)
    return destination


def reconstruct(sha256, specs, scratch, *, min_free_bytes=None):
    """The file with ``sha256`` from its spec: a raw blob, or an exact ZIP delta (artifact_zip_delta, receipt and
    payload blobs) over another spec'd file, recursively (a full base and its delta chain). Every reconstruction is
    verified against its independently bound target hash before it is used. Returns the reconstructed path."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import artifact_zip_delta
    if min_free_bytes is None:
        min_free_bytes = int(os.environ.get('NATLANG_CW_DELTA_MIN_FREE_BYTES', artifact_zip_delta.FLOOR))
    spec = specs[sha256]
    target = scratch / f'file-{sha256}'
    if target.is_file() and target.stat().st_size == spec['bytes'] and file_digest(target) == sha256:
        return target
    if spec['encoding'] == 'raw':
        return _fetch_verified(spec, target)
    if spec['encoding'] != 'zip-delta':
        raise ValueError(f'unknown checkpoint encoding {spec["encoding"]!r}')
    base = reconstruct(spec['base'], specs, scratch, min_free_bytes=min_free_bytes)
    delta = scratch / f'delta-{sha256}'
    receipt = _fetch_verified(spec['receipt'], delta / 'receipt.json')
    name = json.loads(receipt.read_text())['payload']['path']
    if Path(name).name != name:
        raise ValueError('delta payload name is not a plain file name')
    _fetch_verified(spec['payload'], delta / name)
    if target.exists():
        target.unlink()
    started = time.time()
    artifact_zip_delta.verify(base, receipt, expected_sha256=sha256, expected_bytes=spec['bytes'],
                              expected_manifest_sha256=spec['receipt']['sha256'], out=target,
                              min_free_bytes=min_free_bytes)
    shutil.rmtree(delta, ignore_errors=True)
    event(event='reconstructed', sha256=sha256, base=spec['base'], bytes=spec['bytes'],
          seconds=round(time.time() - started, 1))
    return target


def _restore_input(item, scratch):
    """A run input at its DGX path: ``file`` (one exact file), ``archive`` (tar.zst of exact files at their absolute
    paths), ``runtime`` (a frozen runtime archive with ``runtime/`` and ``runtime-manifest.json`` members, installed
    whole at ``path`` after its protected-input closure hash is checked) or ``tree`` (tar.zst of one directory,
    unpacked under ``extract_to``)."""
    kind = item.get('kind', 'tree')
    target = Path(item['path'])
    if kind == 'file':
        _fetch_verified(item, target)
        return
    marker = target.parent / f'.{target.name}.input-sha256'
    if marker.exists() and marker.read_text() == item['sha256'] and (kind == 'archive' or target.exists()):
        return
    temp = _fetch_verified(item, scratch / item['sha256'])
    if kind == 'archive':
        # Unpacked beside the scratch blob, then moved file by file: the DGX roots in a sandbox are symlinks onto the
        # volume, and tar extracting at / through them fails (EXDEV); shutil.move copies across devices when needed.
        unpacked = scratch / f'archive-{item["sha256"]}'
        shutil.rmtree(unpacked, ignore_errors=True)
        unpacked.mkdir(parents=True)
        untar_zst(temp, unpacked)
        for member in item.get('files', []):
            source, destination = unpacked / member['path'].lstrip('/'), Path(member['path'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            shutil.move(str(source), str(destination))
            # The closure tar is deterministic (mtime 0); a restored file gets a fresh mtime so the size+mtime keyed
            # digest cache never reuses a previous restore's digest for same-size new bytes (2026-11-03: a rebound
            # sft-capacity-descriptor.json failed 'resident artifact changed' on its stale cached digest).
            os.utime(destination)
        shutil.rmtree(unpacked, ignore_errors=True)
        for member in item.get('files', []):
            if file_digest(member['path']) != member['sha256']:
                raise ValueError(f'{member["path"]}: restored bytes differ from the closure')
    elif kind == 'runtime':
        stage_dir = target.parent / f'.{target.name}.installing'
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True)
        untar_zst(temp, stage_dir)
        closure = stage_dir / 'runtime' / 'runtime-input-closure.json'
        if item.get('closure_sha256') and file_digest(closure) != item['closure_sha256']:
            raise ValueError(f'{target}: runtime closure differs from the declared frozen runtime')
        if target.exists():
            shutil.rmtree(target)
        os.rename(stage_dir / 'runtime', target)
        shutil.rmtree(stage_dir, ignore_errors=True)
    else:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        untar_zst(temp, item['extract_to'])
    temp.unlink()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(item['sha256'])
    event(event='input_restored', kind=kind, path=str(target), sha256=item['sha256'], bytes=item['bytes'])


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
    # Run inputs first (frozen runtime, data closure, task source): the controller reads them at their DGX paths.
    for item in parent.get('inputs', []):
        _restore_input(item, scratch)
    specs = dict(parent.get('bases', {}))
    for blob in parent['blobs']:
        if blob['encoding'] == 'zip-delta':
            specs[blob['sha256']] = blob
            source = reconstruct(blob['sha256'], specs, scratch)
            target = Path(blob['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
        elif blob['encoding'] == 'raw':
            target = Path(blob['path'])
            _fetch_verified(blob, target)
        else:
            temp = _fetch_verified(blob, scratch / blob['sha256'])
            destination = Path(blob['path']) if blob['kind'] == 'stage' else lineage
            destination.mkdir(parents=True, exist_ok=True)
            untar_zst(temp, destination)
            temp.unlink()
    for link in parent.get('links', []):
        target = Path(link['path'])
        if target.exists() or target.is_symlink():
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        if 'symlink_to' in link:  # a DGX symlink (bound records resolve through it), recreated as one
            os.symlink(link['symlink_to'], target)
        else:
            os.link(link['same_as'], target)
    state = controller(lineage)[0] / 'state.json'
    if parent.get('state_sha256') and file_digest(state) != parent['state_sha256']:
        raise ValueError('restored state.json differs from the export')
    shutil.rmtree(scratch, ignore_errors=True)  # reconstructed bases are only intermediates
    exports = lineage.parent / '.exports' / lineage.name
    exports.mkdir(parents=True, exist_ok=True)
    (exports / 'parent-seq').write_text(str(parent['seq'] if isinstance(parent.get('seq'), int) else -1))
    event(event='parent_restored', seq=parent.get('seq'), trained_rounds=parent.get('trained_rounds'),
          checkpoint=parent.get('checkpoint'))
    return 0


def prune(lineage, exports):
    """Keep every blob and checkpoint named by the newest uploaded export or anything newer, and the active controller's
    own checkpoints; drop older staged exports' blobs and superseded checkpoints of earlier rounds and windows
    (prune_rl.sh semantics)."""
    lineage, exports = Path(os.path.abspath(lineage)), Path(exports)
    uploaded = sorted(int(p.stem.split('.')[0]) for p in exports.glob('*.uploaded'))
    if not uploaded:
        return 0
    newest = uploaded[-1]
    keep, named = set(), set()
    for path in exports.glob('[0-9]*.json'):
        manifest = json.loads(path.read_text())
        if manifest['seq'] >= newest:
            keep |= {b['sha256'] for b in manifest['blobs']}
            named |= {os.path.abspath(p) for p in manifest.get('checkpoints') or [manifest['checkpoint']]}
    removed = 0
    for blob in (exports / 'cas').iterdir():
        if blob.name not in keep and not blob.name.startswith('.'):
            removed += blob.stat().st_size
            blob.unlink()
    if (lineage / 'tune.json').is_file():  # a tune shard's checkpoint is its read-only parent
        event(event='pruned', newest_uploaded=newest, bytes=removed)
        return 0
    out, config_path = controller(lineage)
    config = json.loads(config_path.read_text())
    state, _ = _state(out, config)
    live = {os.path.abspath(p) for p in resume_checkpoints(out, config, state)} | {os.path.abspath(state['checkpoint'])}
    if not live <= named:  # the live boundary is not yet safely exported
        event(event='pruned', newest_uploaded=newest, bytes=removed)
        return 0
    candidates = []
    windows = json.loads((lineage / 'lineage.json').read_text())['windows'] if (lineage / 'lineage.json').is_file() else []
    for window in windows[:-1]:
        candidates.append(Path(window['out']) / 'learner')
    if not windows:
        for path in lineage.iterdir():
            match = ROUND.match(path.name)
            if match and int(match.group(1)) < int(state['round']):
                candidates.append(path / 'learner')
                shutil.rmtree(path / 'actor-store' / 'sglang-hf', ignore_errors=True)
    for directory in candidates:
        for name in ('checkpoint.pt', 'best-checkpoint.pt', 'rl-native-objective.json'):
            target = directory / name
            if target.exists() and os.path.abspath(target) not in live | named:
                removed += target.stat().st_size
                target.unlink()
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
    if command == 'activate':
        activate(*args)
        return 0
    if command == 'progress':
        print(json.dumps(progress(*args)))
        return 0
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
