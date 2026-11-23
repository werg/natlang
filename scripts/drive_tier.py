#!/usr/bin/env python3
"""Google Drive as the third storage tier (NVMe -> HDD -> Drive) and the Colab sessions' shared store.

Everything lives under one Drive folder, ``natlang`` (the ``drive.file`` scope sees only what these tools create):
``natlang/hdd`` mirrors archived HDD files (named by their path under ``/mnt/external``), ``natlang/checkpoints`` holds
checkpoints by sha256 (``<sha>.pt``) for Colab bootstraps, ``natlang/colab/<run>`` holds what Colab learners write
(learner state, factors). Standard library only, so a Colab VM runs it too.

Credentials: on the DGX the ADC user token (``gcloud auth application-default print-access-token``) with the quota
project ``x-goog-user-project`` (``NATLANG_DRIVE_PROJECT``, default awareness-481308); elsewhere ``--token-file`` (a
short-lived access token the DGX writes there: ``push-token``). Tokens are never printed or logged.

Integrity: an upload streams the file once, hashing md5 and sha256; Drive's ``md5Checksum`` and size must equal the
local ones before anything relies on the copy. A download is hashed the same way and lands only whole (``.part`` then
rename). The tier deletes an HDD file only after that check, leaving ``<file>.ondrive.json`` (id, md5, sha256, size) in
its place, and appends every placement to the manifest (``~/data/drive-tier/manifest.jsonl``, also mirrored to
``natlang/manifest.jsonl``), so ``restore PATH`` brings a file or a whole directory back in one command.

    drive_tier.py bench --mb 512                     # upload + download throughput (a random file, then deleted)
    drive_tier.py archive [--apply] [--budget-gb 9]  # one HDD -> Drive pass (rate cap, pauses during Colab transfers)
    drive_tier.py restore /mnt/external/some/dir     # everything archived under that path, back in place
    drive_tier.py put FILE KEY / get KEY DEST        # one object (KEY like checkpoints/<sha>.pt)
    drive_tier.py put-dir DIR PREFIX / get-dir PREFIX DEST   # a directory (learner state, factors), whole
    drive_tier.py push-token HOST PATH               # a fresh access token to HOST:PATH over ssh (0600)
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = 'https://www.googleapis.com/drive/v3'
UPLOAD = 'https://www.googleapis.com/upload/drive/v3/files'
FOLDER_MIME = 'application/vnd.google-apps.folder'
ROOT_NAME = 'natlang'
CHUNK = 16 * 2**20          # multiple of 256 KiB, as resumable uploads require
GIB = 2**30
HDD = Path('/mnt/external')
STATE = Path(os.environ.get('NATLANG_DRIVE_STATE', Path.home() / 'data' / 'drive-tier'))
MANIFEST = STATE / 'manifest.jsonl'
PROJECT = os.environ.get('NATLANG_DRIVE_PROJECT', 'awareness-481308')
STUB_SUFFIX = '.ondrive.json'


def log(**fields):
    print(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%S'), **fields}), flush=True)


# ---------- credentials and requests ----------

ADC = Path(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS',
                          Path.home() / '.config' / 'gcloud' / 'application_default_credentials.json'))


def adc_token() -> str:
    """An access token from the ADC user credentials (refresh-token grant: no gcloud on PATH needed, as in a ledger
    unit); gcloud when the file is not a user credential."""
    try:
        cred = json.loads(ADC.read_text())
    except (OSError, ValueError):
        cred = {}
    if cred.get('type') == 'authorized_user':
        body = urllib.parse.urlencode({k: cred[k] for k in ('client_id', 'client_secret', 'refresh_token')}
                                      | {'grant_type': 'refresh_token'}).encode()
        with urllib.request.urlopen('https://oauth2.googleapis.com/token', body, timeout=60) as resp:
            return json.loads(resp.read())['access_token']
    gcloud = next((g for g in ('gcloud', str(Path.home() / 'google-cloud-sdk/bin/gcloud'))
                   if subprocess.run(['sh', '-c', f'command -v {g}'], capture_output=True).returncode == 0), 'gcloud')
    return subprocess.run([gcloud, 'auth', 'application-default', 'print-access-token'], capture_output=True,
                          text=True, check=True, timeout=120).stdout.strip()

class Drive:
    """A Drive v3 client on one access token, refreshed from gcloud (DGX) or re-read from the token file (VM)."""

    def __init__(self, token_file: str | None = None, project: str | None = PROJECT):
        self.token_file, self.project = token_file, project
        self._token, self._token_at = None, 0.0
        self._folders: dict[str, str] = {}

    def token(self, fresh=False) -> str:
        if self._token and not fresh and time.time() - self._token_at < 600:
            return self._token
        if self.token_file:
            self._token = Path(self.token_file).read_text().strip()
        else:
            self._token = adc_token()
        self._token_at = time.time()
        return self._token

    def request(self, method, url, *, params=None, body=None, headers=None, json_body=None, raw=False, timeout=300):
        if params:
            url += ('&' if '?' in url else '?') + urllib.parse.urlencode(params)
        for attempt in range(6):
            head = {'Authorization': 'Bearer ' + self.token(fresh=attempt > 0)}
            if self.project:      # ADC user tokens need the quota project, on the VM too
                head['x-goog-user-project'] = self.project
            head.update(headers or {})
            data = body
            if json_body is not None:
                data = json.dumps(json_body).encode()
                head['Content-Type'] = 'application/json; charset=UTF-8'
            req = urllib.request.Request(url, data=data, method=method, headers=head)
            try:
                resp = urllib.request.urlopen(req, timeout=timeout)
                return resp if raw else _json(resp.read())
            except urllib.error.HTTPError as error:
                if error.code == 308 and raw:          # resumable upload: chunk accepted, more expected
                    return error
                if error.code in (401, 403, 429, 500, 502, 503, 504) and attempt < 5:
                    if error.code == 403 and b'rateLimit' not in error.read():
                        raise
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError('unreachable')

    # ---------- folders and names ----------

    def find(self, name, parent, folder=False):
        quoted = name.replace('\\', '\\\\').replace("'", "\\'")
        q = f"name = '{quoted}' and '{parent}' in parents and trashed = false"
        if folder:
            q += f" and mimeType = '{FOLDER_MIME}'"
        files = self.request('GET', f'{API}/files', params={
            'q': q, 'fields': 'files(id,name,size,md5Checksum,appProperties)', 'spaces': 'drive'})['files']
        return files[0] if files else None

    def folder(self, path: str) -> str:
        """The id of folder ``natlang/<path>`` ('' for the root), created when missing."""
        if path in self._folders:
            return self._folders[path]
        if not path:
            found = self.request('GET', f'{API}/files', params={
                'q': f"name = '{ROOT_NAME}' and mimeType = '{FOLDER_MIME}' and 'root' in parents and trashed = false",
                'fields': 'files(id)'})['files']
            ident = found[0]['id'] if found else self.request(
                'POST', f'{API}/files', params={'fields': 'id'},
                json_body={'name': ROOT_NAME, 'mimeType': FOLDER_MIME, 'parents': ['root']})['id']
        else:
            parent_path, _, name = path.rpartition('/')
            parent = self.folder(parent_path)
            found = self.find(name, parent, folder=True)
            ident = found['id'] if found else self.request(
                'POST', f'{API}/files', params={'fields': 'id'},
                json_body={'name': name, 'mimeType': FOLDER_MIME, 'parents': [parent]})['id']
        self._folders[path] = ident
        return ident

    def lookup(self, key: str):
        """The Drive file for ``natlang/<key>`` (folder part / file name), or None."""
        folder_path, _, name = key.rpartition('/')
        return self.find(name, self.folder(folder_path))

    # ---------- transfer ----------

    def upload(self, path, key, *, rate=None, pause=None, properties=None) -> dict:
        """Resumable upload of PATH to ``natlang/<key>`` (replacing nothing: an existing name is an error unless its
        md5 matches, then it is reused). Returns {id, bytes, md5, sha256, seconds}; raises unless Drive's md5 and size
        equal the bytes read. RATE caps bytes/second; PAUSE() true holds the transfer between chunks."""
        path = Path(path)
        size = path.stat().st_size
        folder_path, _, name = key.rpartition('/')
        parent = self.folder(folder_path)
        start = self.request('POST', UPLOAD, params={'uploadType': 'resumable', 'fields': 'id'}, raw=True,
                             headers={'X-Upload-Content-Type': 'application/octet-stream',
                                      'X-Upload-Content-Length': str(size)},
                             json_body={'name': name, 'parents': [parent],
                                        'appProperties': {k: str(v)[:100] for k, v in (properties or {}).items()}})
        session = start.headers['Location']
        md5, sha = hashlib.md5(), hashlib.sha256()
        sent, began, result = 0, time.time(), None
        with open(path, 'rb') as stream:
            while sent < size or size == 0:
                while pause and pause():
                    time.sleep(30)
                block = stream.read(CHUNK)
                md5.update(block)
                sha.update(block)
                end = sent + len(block) - 1
                headers = {'Content-Range': f'bytes {sent}-{end}/{size}' if block else f'bytes */{size}',
                           'Content-Type': 'application/octet-stream'}
                for attempt in range(8):
                    try:
                        resp = self.request('PUT', session, body=block, headers=headers, raw=True, timeout=600)
                        break
                    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                        if attempt == 7:
                            raise
                        log(event='upload chunk retry', key=key, offset=sent, error=repr(error)[:120])
                        time.sleep(2 ** attempt)
                sent += len(block)
                if getattr(resp, 'code', None) in (200, 201):
                    result = _json(resp.read())
                    break
                if rate:
                    ahead = sent / rate - (time.time() - began)
                    if ahead > 0:
                        time.sleep(ahead)
                if size == 0:
                    break
        if result is None:
            raise RuntimeError(f'upload of {path} ended without a file id')
        meta = self.request('GET', f'{API}/files/{result["id"]}', params={'fields': 'id,size,md5Checksum'})
        if int(meta.get('size', -1)) != size or meta.get('md5Checksum') != md5.hexdigest():
            self.request('DELETE', f'{API}/files/{result["id"]}', raw=True)
            raise RuntimeError(f'Drive copy of {path} does not verify (size {meta.get("size")} vs {size}, md5 '
                               f'{meta.get("md5Checksum")} vs {md5.hexdigest()}); deleted it')
        return {'id': result['id'], 'bytes': size, 'md5': md5.hexdigest(), 'sha256': sha.hexdigest(),
                'seconds': round(time.time() - began, 1)}

    def download(self, file_id, dest, *, md5=None, sha256=None) -> dict:
        """File FILE_ID to DEST, whole or not at all; checked against Drive's md5 (and SHA256 when given)."""
        dest = Path(dest)
        meta = self.request('GET', f'{API}/files/{file_id}', params={'fields': 'size,md5Checksum'})
        want_md5 = md5 or meta.get('md5Checksum')
        part = dest.with_name(dest.name + '.part')
        dest.parent.mkdir(parents=True, exist_ok=True)
        h_md5, h_sha = hashlib.md5(), hashlib.sha256()
        began = time.time()
        resp = self.request('GET', f'{API}/files/{file_id}', params={'alt': 'media'}, raw=True, timeout=600)
        with open(part, 'wb') as out:
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                h_md5.update(block)
                h_sha.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        got = part.stat().st_size
        if got != int(meta['size']) or h_md5.hexdigest() != want_md5 or (sha256 and h_sha.hexdigest() != sha256):
            part.unlink(missing_ok=True)
            raise RuntimeError(f'download of {file_id} does not verify ({got} bytes, md5 {h_md5.hexdigest()})')
        os.replace(part, dest)
        return {'bytes': got, 'md5': h_md5.hexdigest(), 'sha256': h_sha.hexdigest(),
                'seconds': round(time.time() - began, 1)}

    def delete(self, file_id):
        self.request('DELETE', f'{API}/files/{file_id}', raw=True)


def _json(data):
    return json.loads(data) if data else {}


# ---------- manifest ----------

def manifest_append(record, manifest=None):
    manifest = manifest or MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, 'a') as stream:
        stream.write(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%S'), **record}) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def manifest_latest(manifest=None) -> dict[str, dict]:
    """The latest record per local path."""
    manifest = manifest or MANIFEST
    latest = {}
    if manifest.exists():
        for line in manifest.open():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            latest[record['path']] = record
    return latest


def mirror_manifest(drive: Drive, manifest=None):
    """The manifest's copy on Drive (``natlang/manifest.jsonl``), so restores survive losing the DGX state dir."""
    manifest = manifest or MANIFEST
    old = drive.lookup('manifest.jsonl')
    new = drive.upload(manifest, 'manifest.jsonl.new')
    if old:
        drive.delete(old['id'])
    drive.request('PATCH', f'{API}/files/{new["id"]}', json_body={'name': 'manifest.jsonl'})


# ---------- the HDD -> Drive tier ----------

# natlang's finished-run archives on the HDD; other projects' data (bgkit, hindsight, megatoken, ComfyUI, Vibestudio
# campaigns) is not here and is never touched (owner decision pending)
DEFAULT_ROOTS = (
    'dgx-moved-20261104', 'dgx-moved-20261107-internal', 'dgx-ops-archive-20261118', 'lfm3-rl-dgx-20261029-ops',
    'natlang-grpo-v1', 'natlang-grpo-v2', 'natlang-grpo-v2-archive', 'natlang-grpo-v3', 'natlang-grpo-v4',
    'natlang-grpo-v5', 'natlang-grpo-bench', 'natlang-lfm3-cloud-20261103', 'natlang-lfm3-cloud-corrected-20261103',
    'natlang-diloco-smoke-20261103', 'natlang-independent-recovery-20261101', 'natlang-moved-20261028',
    'natlang-moved-20261029', 'natlang-fallow22-20261110', 'natlang-replay-corrected-1103',
)
NEVER = ('bgkit*', '*hindsight*', 'megatoken*', 'ComfyUI*', '*vibestudio*', 'claude-scratch*', 'lost+found',
         'natlang-test-tmp', 'natlang-development-data')


def colab_transfer_active() -> bool:
    """An rsync or scp to or from a Colab host is running (they share the DGX uplink with Drive uploads)."""
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if argv and os.path.basename(argv[0].decode(errors='replace')) in ('rsync', 'scp') and \
                any(a.startswith(b'colab') and b':' in a for a in argv[1:]):
            return True
    return False


def symlink_targets(roots=(Path.home() / 'data', Path.home() / 'natlang' / 'runs', Path.home())) -> list[str]:  # noqa
    """Where symlinks on the root filesystem point into the HDD: such targets are live paths (archived files with a
    link back, recalled trees' cold parts) and stay on the HDD."""
    found, root_dev = set(), os.stat('/').st_dev
    for root in roots:
        depth = 0 if root == Path.home() else None
        for directory, children, names in os.walk(root):
            try:
                if os.stat(directory).st_dev != root_dev:
                    children[:] = []
                    continue
            except OSError:
                continue
            if depth is not None:     # the home directory itself: its top level only
                children[:] = []
            for name in children + names:
                path = os.path.join(directory, name)
                if os.path.islink(path):
                    target = os.path.realpath(path)
                    if target.startswith(str(HDD) + '/'):
                        found.add(target)
    return sorted(found)


def held_paths() -> set[str]:
    """Paths any process holds open, maps, or works in."""
    held = set()
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            held.add(os.readlink(f'/proc/{pid}/cwd'))
            for fd in os.listdir(f'/proc/{pid}/fd'):
                try:
                    held.add(os.readlink(f'/proc/{pid}/fd/{fd}'))
                except OSError:
                    pass
        except OSError:
            pass
    return held


def registered(path: Path) -> bool:
    """True when a data-registry dataset holds PATH (trainers may read it: it stays)."""
    try:
        repo = Path(__file__).resolve().parents[1]
        sys.path[:0] = [str(repo / 'training' / 'neuralese'), str(repo)]
        from natlang_neuralese.common import data_registry
        if not hasattr(registered, 'reg'):
            registered.reg = data_registry.load()
        ds = registered.reg.find(path)
        return ds is not None and ds.get('admission') != data_registry.INVENTORY
    except Exception:          # a registry that cannot answer keeps the file
        return True


def candidates(roots, *, min_mb, min_age_days, exclude, links, held):
    now = time.time()
    for name in roots:
        top = HDD / name
        if not top.is_dir() or top.is_symlink() or any(fnmatch.fnmatch(name, n) for n in NEVER):
            continue
        for directory, children, files in os.walk(top):
            children[:] = sorted(c for c in children if not os.path.islink(os.path.join(directory, c)))
            for file in sorted(files):
                path = os.path.join(directory, file)
                if file.endswith((STUB_SUFFIX, '.part', '.partial', '.archiving')) or os.path.islink(path):
                    continue
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_size < min_mb * 2**20 or now - max(st.st_mtime, st.st_ctime) < min_age_days * 86400:
                    continue
                if any(path == x or path.startswith(x.rstrip('/') + '/') for x in exclude):
                    continue
                if _linked(path, links) or path in held:
                    continue
                yield Path(path), st


def _linked(path: str, links: set[str]) -> bool:
    """PATH or one of its directories is a symlink target (a live path behind a link)."""
    probe = path
    while probe.startswith(str(HDD) + '/'):
        if probe in links:
            return True
        probe = os.path.dirname(probe)
    return False


def archive(args, drive=None) -> int:
    drive = drive or Drive()
    exclude = [os.path.expanduser(line.strip()) for line in _lines(Path.home() / '.config/natlang/archive-exclude.txt')]
    links, held = set(symlink_targets() if args.links is None else args.links), held_paths()
    rate = args.rate_mb * 2**20 if args.rate_mb else None
    budget, moved, n, skipped_registry = args.budget_gb * GIB, 0, 0, 0
    deadline = time.time() + args.max_minutes * 60
    latest = manifest_latest()
    for path, st in candidates(args.root or DEFAULT_ROOTS, min_mb=args.min_mb, min_age_days=args.min_age_days,
                               exclude=exclude, links=links, held=held):
        if moved + st.st_size > budget or time.time() > deadline:
            break
        if registered(path):
            skipped_registry += 1
            continue
        key = 'hdd/' + str(path.relative_to(HDD))
        if not args.apply:
            log(event='would archive', path=str(path), gb=round(st.st_size / GIB, 2))
            moved += st.st_size
            n += 1
            continue
        record = latest.get(str(path))
        if record and record.get('op') == 'put' and record.get('bytes') == st.st_size \
                and abs(record.get('mtime', 0) - st.st_mtime) < 1:
            done = record          # uploaded and verified before; the delete did not happen
        else:
            done = drive.upload(path, key, rate=rate, pause=colab_transfer_active,
                                properties={'natlang_path': str(path)[-100:]})
            done = {'op': 'put', 'path': str(path), 'key': key, 'mtime': st.st_mtime, **done}
            manifest_append(done)
        # the copy verified; the file is unchanged since it was read; then the HDD copy goes, a stub stays
        now = os.stat(path)
        if now.st_size != st.st_size or now.st_mtime != st.st_mtime:
            log(event='changed while uploading: kept', path=str(path))
            continue
        stub = path.with_name(path.name + STUB_SUFFIX)
        stub.write_text(json.dumps({'drive_id': done['id'], 'key': key, 'bytes': done['bytes'], 'md5': done['md5'],
                                    'sha256': done['sha256'], 'mtime': st.st_mtime,
                                    'restore': f'scripts/drive_tier.py restore {path}'}) + '\n')
        path.unlink()
        manifest_append({'op': 'drop-local', 'path': str(path), 'key': key, 'id': done['id'], 'bytes': done['bytes'],
                         'md5': done['md5'], 'sha256': done['sha256'], 'mtime': st.st_mtime})
        moved += st.st_size
        n += 1
        log(event='archived', path=str(path), gb=round(st.st_size / GIB, 2), seconds=done.get('seconds'))
    if args.apply and n:
        mirror_manifest(drive)
    log(event='summary', applied=args.apply, files=n, gb=round(moved / GIB, 2), skipped_registered=skipped_registry,
        hdd_free_gb=round(os.statvfs(HDD).f_bavail * os.statvfs(HDD).f_frsize / GIB, 1))
    return 0


def restore(args, drive=None) -> int:
    drive = drive or Drive(args.token_file)
    prefix = os.path.abspath(args.path)
    todo = [r for p, r in manifest_latest().items()
            if r.get('op') == 'drop-local' and (p == prefix or p.startswith(prefix.rstrip('/') + '/'))]
    for record in todo:
        path = Path(record['path'])
        got = drive.download(record['id'], path, md5=record['md5'], sha256=record['sha256'])
        os.utime(path, (record['mtime'], record['mtime']))
        path.with_name(path.name + STUB_SUFFIX).unlink(missing_ok=True)
        manifest_append({'op': 'restored', 'path': str(path), 'key': record['key'], 'id': record['id'],
                         'bytes': got['bytes'], 'md5': got['md5'], 'sha256': got['sha256'], 'mtime': record['mtime']})
        log(event='restored', path=str(path), gb=round(got['bytes'] / GIB, 2), seconds=got['seconds'])
    log(event='summary', restored=len(todo))
    return 0


def _lines(path):
    try:
        return [l for l in path.read_text().splitlines() if l.strip() and not l.startswith('#')]
    except OSError:
        return []


# ---------- small commands ----------

def bench(args) -> int:
    drive = Drive(args.token_file)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    src = scratch / f'drive-bench-{os.getpid()}.bin'
    down = scratch / f'drive-bench-{os.getpid()}.down'
    with open(src, 'wb') as out:
        for _ in range(args.mb):
            out.write(os.urandom(2**20))
    try:
        up = drive.upload(src, f'bench/{os.uname().nodename}-{int(time.time())}.bin')
        got = drive.download(up['id'], down, md5=up['md5'], sha256=up['sha256'])
        drive.delete(up['id'])
        mb = args.mb * 2**20 / 1e6
        log(event='bench', host=os.uname().nodename, mb=args.mb, upload_mb_s=round(mb / max(up['seconds'], 1e-3), 2),
            download_mb_s=round(mb / max(got['seconds'], 1e-3), 2), upload_s=up['seconds'], download_s=got['seconds'])
    finally:
        src.unlink(missing_ok=True)
        down.unlink(missing_ok=True)
    return 0


def put(args) -> int:
    drive = Drive(args.token_file)
    existing = drive.lookup(args.key)
    if existing and args.if_absent:
        log(event='present', key=args.key, id=existing['id'], md5=existing.get('md5Checksum'))
        return 0
    done = drive.upload(args.file, args.key, rate=args.rate_mb * 2**20 if args.rate_mb else None)
    if existing:              # a replacement (learner state): the new copy verified first, then the old one goes
        drive.delete(existing['id'])
    log(event='put', key=args.key, **done)
    return 0


def get(args) -> int:
    drive = Drive(args.token_file)
    found = drive.lookup(args.key)
    if not found:
        log(event='absent', key=args.key)
        return 3
    got = drive.download(found['id'], args.dest, sha256=args.sha256)
    log(event='got', key=args.key, **got)
    return 0


def put_dir(args) -> int:
    """Every file of DIR under ``natlang/<prefix>/`` (one level of subdirectories kept in the name), each replacing
    its older copy only after the new one verified, then ``complete.json`` (file -> md5) last: a reader takes the set
    only when ``complete.json`` lists exactly what it finds."""
    drive, root = Drive(args.token_file), Path(args.dir)
    done, began = {}, time.time()
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        rel = str(path.relative_to(root)).replace('/', '__')
        key = f'{args.prefix}/{rel}'
        old = drive.lookup(key)
        if old and old.get('md5Checksum') == _md5(path):
            done[rel] = old['md5Checksum']
            continue
        got = drive.upload(path, key)
        if old:
            drive.delete(old['id'])
        done[rel] = got['md5']
    marker = root.parent / f'.{root.name}.complete.json'
    marker.write_text(json.dumps({'files': done, 'at': time.time()}))
    old = drive.lookup(f'{args.prefix}/complete.json')
    drive.upload(marker, f'{args.prefix}/complete.json')
    if old:
        drive.delete(old['id'])
    marker.unlink(missing_ok=True)
    log(event='put-dir', prefix=args.prefix, files=len(done), seconds=round(time.time() - began, 1))
    return 0


def get_dir(args) -> int:
    """``natlang/<prefix>/`` (as ``put-dir`` wrote it) into DEST, whole: into DEST.part, checked against
    ``complete.json``, then renamed over DEST."""
    drive, dest = Drive(args.token_file), Path(args.dest)
    folder = drive.folder(args.prefix)
    files = drive.request('GET', f'{API}/files', params={
        'q': f"'{folder}' in parents and trashed = false", 'fields': 'files(id,name,md5Checksum)',
        'pageSize': 1000})['files']
    by_name = {f['name']: f for f in files}
    if 'complete.json' not in by_name:
        log(event='absent', prefix=args.prefix)
        return 3
    part = dest.with_name(dest.name + '.part')
    import shutil
    shutil.rmtree(part, ignore_errors=True)
    complete = part / '.complete.json'
    drive.download(by_name['complete.json']['id'], complete)
    want = json.loads(complete.read_text())['files']
    complete.unlink()
    for rel, md5 in want.items():
        found = by_name.get(rel)
        if not found or found.get('md5Checksum') != md5:
            raise RuntimeError(f'{args.prefix}/{rel} is missing or changed since complete.json')
        drive.download(found['id'], part / rel.replace('__', '/'), md5=md5)
    old = dest.with_name(dest.name + '.old')
    shutil.rmtree(old, ignore_errors=True)
    if dest.exists():
        os.replace(dest, old)
    os.replace(part, dest)
    shutil.rmtree(old, ignore_errors=True)
    log(event='get-dir', prefix=args.prefix, files=len(want))
    return 0


def _md5(path) -> str:
    digest = hashlib.md5()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def push_token(args) -> int:
    """A fresh access token to HOST:PATH, through ssh stdin (never on a command line), mode 0600."""
    token = Drive().token(fresh=True)
    remote = (f'umask 077 && mkdir -p "$(dirname {_q(args.path)})" && cat > {_q(args.path)}.tmp && '
              f'mv {_q(args.path)}.tmp {_q(args.path)}')
    subprocess.run(['ssh', args.host, remote], input=token, text=True, check=True, timeout=120,
                   capture_output=True)
    log(event='token pushed', host=args.host, path=args.path)
    return 0


def _q(text):
    import shlex
    return shlex.quote(str(text))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--token-file', help='read the access token from this file (a Colab VM)')
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bench')
    b.add_argument('--mb', type=int, default=512)
    b.add_argument('--scratch', default=str(STATE / 'bench'))
    a = sub.add_parser('archive')
    a.add_argument('--apply', action='store_true')
    a.add_argument('--root', action='append', help='HDD top-level directory (default: natlang run archives)')
    a.add_argument('--budget-gb', type=float, default=9.0)
    a.add_argument('--max-minutes', type=float, default=50.0)
    a.add_argument('--rate-mb', type=float, default=3.0, help='MiB/s cap (0: none)')
    a.add_argument('--min-mb', type=float, default=64.0)
    a.add_argument('--min-age-days', type=float, default=5.0)
    a.set_defaults(links=None)
    r = sub.add_parser('restore')
    r.add_argument('path')
    p = sub.add_parser('put')
    p.add_argument('file')
    p.add_argument('key')
    p.add_argument('--if-absent', action='store_true')
    p.add_argument('--rate-mb', type=float, default=0.0)
    g = sub.add_parser('get')
    g.add_argument('key')
    g.add_argument('dest')
    g.add_argument('--sha256')
    pd = sub.add_parser('put-dir')
    pd.add_argument('dir')
    pd.add_argument('prefix')
    gd = sub.add_parser('get-dir')
    gd.add_argument('prefix')
    gd.add_argument('dest')
    t = sub.add_parser('push-token')
    t.add_argument('host')
    t.add_argument('path')
    args = parser.parse_args(argv)
    return {'bench': bench, 'archive': archive, 'restore': restore, 'put': put, 'get': get,
            'put-dir': put_dir, 'get-dir': get_dir, 'push-token': push_token}[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
