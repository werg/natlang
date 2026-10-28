#!/usr/bin/env python3
"""Content-addressed input/output bundles in CoreWeave AI Object Storage (S3-compatible).

A bundle manifest (schema ``natlang.coreweave-bundle/1``) lists every file a job needs: the path where it is
materialized (repo-relative, or absolute for model weights and archived checkpoints), its SHA-256, its size and its
object key ``cas/sha256/<2 hex>/<64 hex>``. Objects are keyed by content, so one copy serves every bundle and
every sandbox, re-uploads are no-ops, and a hard link or exact duplicate costs nothing. The manifest itself is stored
under ``bundles/<sha256 of its canonical JSON>.json``: a bundle id is a content hash, like the corpus manifests under
training/corpus-manifests/, whose recorded hashes it reuses instead of re-reading 10 GB files.

Arrival is verified: ``fetch`` hashes every downloaded file before moving it into place and refuses a mismatch, and
writes a receipt. Outputs travel the same way back: ``publish`` builds a manifest of a directory (a stage's --out),
uploads it and names it under the job's output prefix; DGX or Pop ``fetch``es that manifest (optionally remapping a
path prefix). Copies are not training admission (AGENTS.md): a bundle moves bytes, the recipe's pinned hashes decide.

  plan     --recipe R [--stage S ...] [--corpus ID ...] [--extra PATH ...] [--prebuilt NAME ...] [--code-sha SHA]
           -o manifest.json
  upload   manifest.json --bucket B [--dry-run]
  fetch    --bucket B (--manifest-key K | --pointer-key outputs/<job>/<dir>/latest.json | --manifest FILE) [--map OLD=NEW ...] [--dry-run]
  publish  DIR --bucket B --prefix outputs/<job>/ [--dry-run]     pointer: <prefix><DIR name>/latest.json
  watch    DIR... --bucket B --prefix outputs/<job>/ [--interval 60] [--max-interval 1800]
  verify   manifest.json [--map OLD=NEW ...]
  ping     --bucket B [--prefix P]
  gc       --bucket B [--drop-job JOB ...] [--keep-bundle ID ...] [--out plan.json] [--execute]

Credentials: never in files or arguments. Inside a sandbox the AWS SDK reads the sandbox's injected Workload Identity
Federation token (AWS_CONTAINER_CREDENTIALS_FULL_URI etc.); on DGX/Pop the CoreWeave API token from
COREWEAVE_API_TOKEN or CWSANDBOX_API_KEY is exchanged for temporary credentials through the documented
api-token endpoint. Requires boto3 >= 1.42.5 (older versions reject the non-loopback credentials host).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex, sha256_hex  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex_cached  # noqa: E402

SCHEMA = 'natlang.coreweave-bundle/1'
DEFAULT_ENDPOINT = 'https://cwobject.com'
TOKEN_EXCHANGE_URI = 'https://api.coreweave.com/v1/cwobject/temporary-credentials/api-token'
# Not content: free-space reservations, the trainers' atomic-write temp names, our own fetch temps.
SKIP_SUFFIXES = ('.reserve', '.pending', '.pending-link', '.tmp', '.partial')
# Files up to this size are read into memory once, so the bytes hashed are the bytes uploaded even while a log grows;
# larger files (checkpoints, written by atomic rename) are hashed and uploaded through one open descriptor.
SMALL_FILE_BYTES = 64 * 1024 * 1024
# A change of a file matching one of these names starts an incremental publish (watch).
DEFAULT_TRIGGERS = ('*.pt', 'report.json', '*certificate*.json')


class BundleError(ValueError):
    pass


def object_key(sha256: str) -> str:
    if len(sha256) != 64 or any(c not in '0123456789abcdef' for c in sha256):
        raise BundleError(f'not a lowercase sha256 hex digest: {sha256!r}')
    return f'cas/sha256/{sha256[:2]}/{sha256}'


def manifest_id(manifest: dict[str, Any]) -> str:
    return canonical_json_sha256_hex({k: v for k, v in manifest.items() if k not in ('id', 'created', 'directory')})


def manifest_key(manifest: dict[str, Any]) -> str:
    return f'bundles/{manifest["id"]}.json'


def _safe_relative(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute():
        if '..' in pure.parts:
            raise BundleError(f'unsafe path: {path}')
        return str(pure)
    if '..' in pure.parts or not pure.parts:
        raise BundleError(f'unsafe relative path: {path}')
    return str(pure)


def entry(path: str, source: Path, sha256: str | None = None) -> dict[str, Any]:
    """One manifest file entry; ``sha256`` given (pinned by a recipe or corpus manifest) or computed (cached)."""
    if not source.is_file():
        raise BundleError(f'missing input file: {source} (for {path})')
    digest = sha256 or sha256_file_hex_cached(source)
    return {'path': _safe_relative(path), 'sha256': digest, 'bytes': source.stat().st_size, 'key': object_key(digest)}


def recipe_entries(recipe_path: Path, stages: Iterable[str] = (), repo: Path = REPO) -> list[tuple[dict, Path]]:
    """Every input binding of the resolved recipe (or only those the named stages read), with its pinned sha256."""
    from natlang_neuralese.train.recipe import resolve_recipe_data
    recipe = resolve_recipe_data(recipe_path)
    bindings = recipe.get('input_bindings', {})
    wanted = set(bindings)
    if stages:
        known = {stage['id']: stage for stage in recipe['stages']}
        missing = [s for s in stages if s not in known]
        if missing:
            raise BundleError(f'recipe {recipe["id"]} has no stage {", ".join(missing)}; stages: {", ".join(known)}')
        wanted = {role for s in stages for role in (known[s].get('inputs') or {}).values() if role in bindings}
    out = []
    for role in sorted(wanted):
        value = bindings[role]
        path, pinned = (value, None) if isinstance(value, str) else (value['path'], value.get('sha256'))
        source = Path(path) if Path(path).is_absolute() else repo / path
        out.append((entry(path, source, pinned), source))
    return out


def corpus_entries(corpus_id: str, repo: Path = REPO) -> list[tuple[dict, Path]]:
    """Files of a published corpus snapshot (training/corpus-manifests/<id>.json), with their recorded hashes."""
    manifest = json.loads((repo / 'training' / 'corpus-manifests' / f'{corpus_id}.json').read_text())
    base = manifest['path']
    out = []
    for item in manifest['files']:
        path = f'{base}/{item["path"]}'
        source = repo / path
        out.append((entry(path, source, item['sha256']), source))
    return out


def extra_entries(path: str, repo: Path = REPO) -> list[tuple[dict, Path]]:
    """A file or a directory (recursively; symlinked files are followed), repo-relative or absolute."""
    source = Path(path) if Path(path).is_absolute() else repo / path
    if source.is_file():
        return [(entry(path, source), source)]
    if not source.is_dir():
        raise BundleError(f'missing extra input: {source}')
    out = []
    for child in sorted(p for p in source.rglob('*') if p.is_file()):
        logical = str(PurePosixPath(path) / child.relative_to(source).as_posix())
        out.append((entry(logical, child), child))
    return out


def hf_entries(spec: str) -> list[tuple[dict, Path | None]]:
    """``REPO@REVISION=LOCAL_DIR``: every file of a public Hugging Face repo revision as a URL entry at LOCAL_DIR.

    LFS files carry their sha256 in the HF tree (the oid); other files are hashed from the local copy, which must be
    the same revision (a hub download). The sandbox downloads from HF and verifies; nothing is uploaded.
    """
    import urllib.request
    repo_rev, local = spec.split('=', 1)
    repo, revision = repo_rev.split('@', 1)
    url = f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        tree = json.loads(response.read())
    out = []
    for node in tree:
        if node.get('type') != 'file':
            continue
        local_file = Path(local) / node['path']
        digest = (node.get('lfs') or {}).get('oid')
        if digest is None:
            if not local_file.is_file():
                raise BundleError(f'{local_file}: the local copy is needed to hash non-LFS file {node["path"]}')
            digest = sha256_file_hex(local_file)
        item = {'path': _safe_relative(str(local_file)), 'sha256': digest, 'bytes': node['size'],
                'key': object_key(digest), 'url': f'https://huggingface.co/{repo}/resolve/{revision}/{node["path"]}'}
        out.append((item, None))
    return out


def code_archive(sha: str, out_dir: Path, repo: Path = REPO) -> tuple[dict, Path]:
    """`git archive` of a commit (the same frozen-snapshot practice as the DGX launchers), content-addressed."""
    resolved = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--verify', f'{sha}^{{commit}}'],
                              capture_output=True, text=True, check=True).stdout.strip()
    target = out_dir / f'natlang-{resolved}.tar.gz'
    with target.open('wb') as stream:
        subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar.gz', resolved], stdout=stream, check=True)
    digest = sha256_file_hex(target)
    return {'git_sha': resolved, 'sha256': digest, 'bytes': target.stat().st_size, 'key': object_key(digest)}, target


def build_manifest(pairs: list[tuple[dict, Path]], code: dict | None = None, note: str = '') -> dict[str, Any]:
    files: dict[str, dict] = {}
    for item, _ in pairs:
        previous = files.get(item['path'])
        if previous and previous['sha256'] != item['sha256']:
            raise BundleError(f'two different contents for {item["path"]}')
        files[item['path']] = item
    manifest = {'schema': SCHEMA, 'note': note, 'code': code, 'files': [files[p] for p in sorted(files)],
                'bytes': sum(f['bytes'] for f in files.values())}
    manifest['id'] = manifest_id(manifest)
    manifest['created'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    return manifest


def check_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get('schema') != SCHEMA:
        raise BundleError(f'not a {SCHEMA} manifest')
    if manifest.get('id') != manifest_id(manifest):
        raise BundleError('manifest id does not match its content (edited or truncated)')
    for item in manifest['files']:
        _safe_relative(item['path'])
        if item['key'] != object_key(item['sha256']):
            raise BundleError(f'object key does not match the sha256 of {item["path"]}')


def remap(path: str, mapping: list[tuple[str, str]]) -> str:
    for old, new in mapping:
        if path == old or path.startswith(old.rstrip('/') + '/'):
            return new.rstrip('/') + path[len(old.rstrip('/')):]
    return path


def target_path(item_path: str, root: Path, mapping: list[tuple[str, str]]) -> Path:
    mapped = remap(item_path, mapping)
    return Path(mapped) if PurePosixPath(mapped).is_absolute() else root / mapped


# ---- object storage (crisp mechanism) ----

def s3_client():
    """boto3 S3 client for AI Object Storage: virtual-hosted addressing is required (path style is refused)."""
    import boto3
    from botocore.client import Config
    token = os.environ.get('COREWEAVE_API_TOKEN') or os.environ.get('CWSANDBOX_API_KEY')
    if 'AWS_CONTAINER_CREDENTIALS_FULL_URI' not in os.environ and token and 'AWS_ACCESS_KEY_ID' not in os.environ:
        os.environ['AWS_CONTAINER_CREDENTIALS_FULL_URI'] = TOKEN_EXCHANGE_URI
        os.environ['AWS_CONTAINER_AUTHORIZATION_TOKEN'] = f'Bearer {token}'
    region = os.environ.get('AWS_REGION') or os.environ.get('NATLANG_CW_REGION')
    if not region:
        raise BundleError('set AWS_REGION (or NATLANG_CW_REGION) to the bucket availability zone, e.g. US-EAST-04A')
    # NATLANG_CW_S3_ADDRESSING=path only for the local S3 stand-in of the rehearsal (rehearse.py).
    addressing = os.environ.get('NATLANG_CW_S3_ADDRESSING', 'virtual')
    return boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL_S3', DEFAULT_ENDPOINT),
                        config=Config(region_name=region, s3={'addressing_style': addressing}))


def _exists(s3, bucket: str, key: str, size: int) -> bool:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # botocore ClientError 404 / NoSuchKey
        if getattr(error, 'response', {}).get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
            return False
        raise
    return head.get('ContentLength') == size


def upload(manifest: dict[str, Any], sources: dict[str, Any], bucket: str, s3, dry_run: bool = False,
           log=print) -> dict[str, int]:
    """Upload missing objects (content-addressed, so existing keys are skipped), then the manifest."""
    check_manifest(manifest)
    stats = {'uploaded': 0, 'skipped': 0, 'bytes_uploaded': 0}
    items = list(manifest['files']) + ([manifest['code']] if manifest.get('code') else [])
    seen = set()
    for item in items:
        if item['key'] in seen:
            continue
        seen.add(item['key'])
        if not dry_run and _exists(s3, bucket, item['key'], item['bytes']):
            stats['skipped'] += 1
            continue
        source = sources[item['key']]
        shown = getattr(source, 'name', None) or ('<memory>' if isinstance(source, bytes) else str(source))
        log(json.dumps({'event': 'upload', 'key': item['key'], 'bytes': item['bytes'], 'from': str(shown),
                        'dry_run': dry_run}))
        if not dry_run:
            if isinstance(source, bytes):
                s3.put_object(Bucket=bucket, Key=item['key'], Body=source)
            elif hasattr(source, 'read'):
                source.seek(0)
                s3.upload_fileobj(source, bucket, item['key'])
            else:
                s3.upload_file(str(source), bucket, item['key'])
        stats['uploaded'] += 1
        stats['bytes_uploaded'] += item['bytes']
    if not dry_run:
        s3.put_object(Bucket=bucket, Key=manifest_key(manifest), Body=json.dumps(manifest, indent=1).encode())
    log(json.dumps({'event': 'manifest', 'key': manifest_key(manifest), 'dry_run': dry_run, **stats}))
    return stats


def download_url(url: str, target: Path, chunk: int = 8 * 1024 * 1024, attempts: int = 4) -> None:
    """HTTP(S) download with a few resumable retries (Range); the caller verifies the sha256."""
    import urllib.request
    for attempt in range(attempts):
        have = target.stat().st_size if target.exists() else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, target.open('ab' if have else 'wb') as out:
                if have and response.status != 206:  # server ignored the range: start over
                    out.truncate(0)
                for block in iter(lambda: response.read(chunk), b''):
                    out.write(block)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * 2 ** attempt)


def fetch(manifest: dict[str, Any], bucket: str | None, s3, root: Path, mapping: list[tuple[str, str]] = (),
          dry_run: bool = False, log=print, inbox: Path | None = None) -> dict[str, Any]:
    """Materialize every file, verifying each sha256 before it is moved into place; write a receipt.

    Sources, first available wins: ``inbox/<sha256>`` (pushed through the sandbox file API by cwctl), the entry's
    public ``url`` (e.g. Hugging Face resolve URLs), then the bucket.
    """
    check_manifest(manifest)
    mapping = list(mapping)
    fetched, present = 0, 0
    used_inbox: set[Path] = set()
    for item in manifest['files']:
        target = target_path(item['path'], root, mapping)
        if target.is_file() and target.stat().st_size == item['bytes'] and sha256_file_hex(target) == item['sha256']:
            present += 1
            continue
        log(json.dumps({'event': 'fetch', 'key': item['key'], 'to': str(target), 'bytes': item['bytes'],
                        'dry_run': dry_run}))
        if dry_run:
            fetched += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.cw-fetch-', delete=False) as tmp:
            temp = Path(tmp.name)
        try:
            pushed = inbox / item['sha256'] if inbox is not None else None
            if pushed is not None and pushed.is_file():
                temp.unlink()
                try:
                    os.link(pushed, temp)  # a link, not a move: duplicate contents share one pushed object
                except OSError:  # another file system (e.g. a target outside the scratch volume)
                    shutil.copyfile(pushed, temp)
                used_inbox.add(pushed)
            elif item.get('url'):
                download_url(item['url'], temp)
            elif s3 is not None and bucket:
                s3.download_file(bucket, item['key'], str(temp))
            else:
                raise BundleError(f'{item["path"]}: no source (not pushed, no url, no bucket)')
            actual = sha256_file_hex(temp)
            if actual != item['sha256']:
                raise BundleError(f'{item["path"]}: downloaded sha256 {actual} != manifest {item["sha256"]}')
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        fetched += 1
    for pushed in used_inbox:  # the placed files hold the bytes now; free the inbox links
        pushed.unlink(missing_ok=True)
    receipt = {'schema': 'natlang.coreweave-fetch-receipt/1', 'bundle': manifest['id'], 'bucket': bucket,
               'fetched': fetched, 'present': present, 'files': len(manifest['files']), 'verified': not dry_run,
               'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if not dry_run:
        receipts = root / '.coreweave' / 'receipts'
        receipts.mkdir(parents=True, exist_ok=True)
        (receipts / f'{manifest["id"]}.json').write_text(json.dumps(receipt, indent=1) + '\n')
    log(json.dumps({'event': 'fetched', **receipt}))
    return receipt


def verify(manifest: dict[str, Any], root: Path, mapping: list[tuple[str, str]] = ()) -> list[str]:
    """Paths whose local bytes differ from the manifest (empty list: all present and exact)."""
    check_manifest(manifest)
    bad = []
    for item in manifest['files']:
        target = target_path(item['path'], root, list(mapping))
        if not target.is_file() or target.stat().st_size != item['bytes'] or sha256_file_hex(target) != item['sha256']:
            bad.append(item['path'])
    return bad


def _skipped(path: Path) -> bool:
    return path.name.endswith(SKIP_SUFFIXES) or path.name.startswith('.cw-fetch-')


def snapshot_file(path: Path, cache: dict) -> tuple[str, int, Any]:
    """(sha256, size, source) of one consistent version of a file.

    Small files are read once into memory. A large file stays open: an atomic rename that replaces it during the
    upload leaves this descriptor on the old inode, so hash and uploaded bytes agree. ``cache`` maps
    (device, inode, size, mtime_ns) to a digest, so an unchanged checkpoint is not rehashed on every publish.
    """
    stream = path.open('rb')
    info = os.fstat(stream.fileno())
    if info.st_size <= SMALL_FILE_BYTES:
        data = stream.read()
        stream.close()
        return sha256_hex(data), len(data), data
    identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if identity not in cache:
        cache[identity] = sha256_file_hex(f'/proc/self/fd/{stream.fileno()}')
    return cache[identity], info.st_size, stream


def close_sources(sources: dict[str, Any]) -> None:
    for source in sources.values():
        if hasattr(source, 'close'):
            source.close()


def directory_manifest(directory: Path, note: str = '', cache: dict | None = None) -> tuple[dict, dict[str, Any]]:
    """Manifest of a directory tree under its absolute path (a stage --out), with snapshot sources for upload.

    The caller closes the sources (close_sources). Files that vanish between listing and opening are skipped.
    """
    # abspath, not resolve: in a sandbox the DGX roots are symlinks into scratch, and the manifest must record the
    # logical path (the one recipes and checkpoints name), not the scratch location behind it.
    directory = Path(os.path.abspath(directory))
    cache = {} if cache is None else cache
    files, sources = [], {}
    for child in sorted(p for p in directory.rglob('*') if p.is_file() and not _skipped(p)):
        try:
            digest, size, source = snapshot_file(child, cache)
        except FileNotFoundError:
            continue
        item = {'path': _safe_relative(str(child)), 'sha256': digest, 'bytes': size, 'key': object_key(digest)}
        files.append((item, None))
        if item['key'] in sources:
            if hasattr(source, 'close'):
                source.close()
        else:
            sources[item['key']] = source
    return build_manifest(files, note=note), sources


def output_prefix(prefix: str, directory: Path) -> str:
    """outputs/<job>/<directory name>/: where one output directory's manifests and pointer live."""
    return f'{prefix.rstrip("/")}/{directory.name}/'


def read_json(s3, bucket: str, key: str) -> dict | None:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)['Body'].read())
    except Exception as error:
        if getattr(error, 'response', {}).get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
            return None
        raise


def publish_directory(directory: Path, bucket: str, prefix: str, s3, cache: dict | None = None, note: str = '',
                      dry_run: bool = False, log=print) -> dict[str, Any]:
    """Upload one consistent snapshot of ``directory``, then move ``<prefix><name>/latest.json`` to it.

    Order makes a reader safe at every moment: objects, then the manifest, then the pointer. The pointer keeps the
    previous manifest key, so a resume can fall back one publish (and gc keeps both).
    """
    manifest, sources = directory_manifest(directory, note, cache)
    try:
        stats = upload(manifest, sources, bucket, s3, dry_run, log)
    finally:
        close_sources(sources)
    pointer_key = output_prefix(prefix, directory) + 'latest.json'
    previous = None if dry_run else read_json(s3, bucket, pointer_key)
    pointer = {'schema': 'natlang.coreweave-output-pointer/1', 'bundle': manifest['id'],
               'manifest_key': manifest_key(manifest), 'directory': os.path.abspath(directory),
               'files': len(manifest['files']), 'bytes': manifest['bytes'], 'at': manifest['created'],
               'previous': previous['manifest_key'] if previous and previous['manifest_key'] != manifest_key(manifest)
               else (previous or {}).get('previous')}
    if not dry_run:
        s3.put_object(Bucket=bucket, Key=pointer_key, Body=json.dumps(pointer, indent=1).encode())
    log(json.dumps({'event': 'published', 'pointer': pointer_key, **pointer, **stats, 'dry_run': dry_run}))
    return pointer


def trigger_signature(directory: Path, triggers: Iterable[str]) -> tuple:
    """(path, inode, size, mtime_ns) of every trigger file: it changes when a checkpoint or report is (re)written."""
    import fnmatch
    if not directory.is_dir():
        return ()
    out = []
    for child in sorted(directory.rglob('*')):
        if child.is_file() and not _skipped(child) and any(fnmatch.fnmatch(child.name, t) for t in triggers):
            try:
                info = child.stat()
            except FileNotFoundError:
                continue
            out.append((str(child), info.st_ino, info.st_size, info.st_mtime_ns))
    return tuple(out)


def watch(directories: list[Path], bucket: str, prefix: str, s3, interval: float = 60, max_interval: float = 1800,
          triggers: Iterable[str] = DEFAULT_TRIGGERS, rounds: int | None = None, sleep=time.sleep, clock=time.monotonic,
          log=print) -> int:
    """Publish each directory whenever a trigger file changes, and at least every ``max_interval`` seconds.

    Runs beside the job (bootstrap.sh), so a sandbox killed at lifetime expiry (no grace period) loses at most the
    work since its last checkpoint. ``rounds`` bounds the loop for tests; the bootstrap stops it with SIGTERM.
    Returns the number of publishes.
    """
    triggers = tuple(triggers)
    cache: dict = {}
    seen: dict[Path, tuple] = {}
    last: dict[Path, float] = {}
    published = 0
    done = 0
    while rounds is None or done < rounds:
        for directory in directories:
            if not directory.is_dir():
                continue
            signature = trigger_signature(directory, triggers)
            changed = signature != seen.get(directory, ())
            due = directory in last and clock() - last[directory] >= max_interval
            if (changed and signature) or due:
                try:
                    publish_directory(directory, bucket, prefix, s3, cache, note='incremental', log=log)
                    published += 1
                    seen[directory] = signature
                    last[directory] = clock()
                except Exception as error:  # a failed round is retried next round; the job is never disturbed
                    log(json.dumps({'event': 'publish_failed', 'directory': str(directory), 'error': repr(error)}))
        done += 1
        if rounds is None or done < rounds:
            sleep(interval)
    return published


def list_all(s3, bucket: str, prefix: str = '') -> list[dict]:
    objects, token = [], None
    while True:
        page = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, **({'ContinuationToken': token} if token else {}))
        objects += page.get('Contents', [])
        if not page.get('IsTruncated'):
            return objects
        token = page['NextContinuationToken']


def _age_hours(item: dict, now: float) -> float:
    modified = item.get('LastModified')
    return float('inf') if modified is None else (now - modified.timestamp()) / 3600


def gc_plan(s3, bucket: str, drop_jobs: Iterable[str] = (), keep_bundles: Iterable[str] = (),
            keep_input_days: float = 14, min_age_hours: float = 24, now: float | None = None) -> dict[str, Any]:
    """Which keys to delete (AGENTS.md storage hygiene, applied to the bucket). Nothing is deleted here.

    Kept roots: every output pointer (outputs/<job>/<dir>/latest.json) of a job not in ``drop_jobs``, with its
    latest and previous manifests; input bundles named in ``keep_bundles`` or written within ``keep_input_days``.
    Everything a kept manifest references (files and code snapshot) is kept. Deleted: the whole outputs/<job>/ prefix
    of each dropped job, manifests no root keeps, and content objects no kept manifest references. Content objects
    and manifests younger than ``min_age_hours`` are never deleted: an upload in flight writes its objects before its
    manifest and pointer.
    """
    now = time.time() if now is None else now
    drop_jobs, keep_bundles = set(drop_jobs), set(keep_bundles)
    outputs = list_all(s3, bucket, 'outputs/')
    manifests = {o['Key']: o for o in list_all(s3, bucket, 'bundles/')}
    content = list_all(s3, bucket, 'cas/')
    delete, keep_manifests = [], set()
    for item in outputs:
        job = item['Key'].split('/')[1]
        if job in drop_jobs:
            delete.append(item)
        elif item['Key'].endswith('/latest.json'):
            pointer = read_json(s3, bucket, item['Key']) or {}
            keep_manifests.update(k for k in (pointer.get('manifest_key'), pointer.get('previous')) if k)
    for key, item in manifests.items():
        bundle_id = key.removeprefix('bundles/').removesuffix('.json')
        if bundle_id in keep_bundles or _age_hours(item, now) < max(keep_input_days * 24, min_age_hours):
            keep_manifests.add(key)
    referenced = set()
    for key in sorted(keep_manifests):
        manifest = read_json(s3, bucket, key)
        if manifest is None:
            continue
        referenced.update(f['key'] for f in manifest['files'])
        if manifest.get('code'):
            referenced.add(manifest['code']['key'])
    delete += [item for key, item in manifests.items() if key not in keep_manifests]
    delete += [item for item in content if item['Key'] not in referenced and _age_hours(item, now) >= min_age_hours]
    return {'schema': 'natlang.coreweave-gc-plan/1', 'bucket': bucket, 'drop_jobs': sorted(drop_jobs),
            'kept_manifests': len(keep_manifests), 'kept_objects': len(referenced),
            'delete': [{'key': o['Key'], 'bytes': o.get('Size', 0)} for o in delete],
            'delete_bytes': sum(o.get('Size', 0) for o in delete)}


def gc_execute(s3, bucket: str, plan: dict[str, Any]) -> int:
    keys = [item['key'] for item in plan['delete']]
    for start in range(0, len(keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={'Objects': [{'Key': k} for k in keys[start:start + 1000]],
                                                 'Quiet': True})
    return len(keys)


def _use_client_venv() -> None:
    """Re-execute under the client venv (scripts/coreweave/setup_client.sh) when this boto3 is missing or too old."""
    try:
        import boto3
        version = tuple(int(part) for part in boto3.__version__.split('.')[:2])
    except ImportError:
        version = (0, 0)
    venv = Path(os.environ.get('NATLANG_CW_VENV', Path.home() / '.local/share/natlang/cw-venv')) / 'bin' / 'python'
    if version < (1, 42) and venv.is_file() and Path(sys.executable).resolve() != venv.resolve():
        os.execv(str(venv), [str(venv), *sys.argv])
    if version < (1, 42):
        raise BundleError('boto3 >= 1.42.5 required (container credentials on api.coreweave.com): pip install -U boto3')


def _mapping(values: list[str]) -> list[tuple[str, str]]:
    out = []
    for value in values or []:
        if '=' not in value:
            raise BundleError(f'--map takes OLD=NEW, got {value!r}')
        old, new = value.split('=', 1)
        out.append((old, new))
    return out


def _load_manifest(args, s3) -> dict[str, Any]:
    if args.manifest:
        return json.loads(Path(args.manifest).read_text())
    if args.pointer_key:
        pointer = read_json(s3, args.bucket, args.pointer_key)
        if pointer is None:
            return None
        args.manifest_key = pointer['manifest_key']
    body = s3.get_object(Bucket=args.bucket, Key=args.manifest_key)['Body'].read()
    return json.loads(body)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan', help='build a bundle manifest of a stage\'s inputs')
    p.add_argument('--recipe', type=Path)
    p.add_argument('--stage', action='append', default=[])
    p.add_argument('--corpus', action='append', default=[], help='corpus snapshot id (training/corpus-manifests)')
    p.add_argument('--extra', action='append', default=[], help='file or directory, repo-relative or absolute')
    p.add_argument('--hf', action='append', default=[], metavar='REPO@REV=LOCAL_DIR',
                   help='public Hugging Face files as verified URL entries materialized at LOCAL_DIR')
    p.add_argument('--code-sha', help='commit to package as the code snapshot (git archive)')
    p.add_argument('--prebuilt', action='append', default=[], metavar='NAME',
                   help='prebuilt artifact from scripts/coreweave/prebuilt-artifacts.json (prebuilt.py)')
    p.add_argument('--note', default='')
    p.add_argument('-o', '--out', type=Path, required=True)
    p = sub.add_parser('upload')
    p.add_argument('manifest', type=Path)
    p.add_argument('--bucket', required=True)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('fetch')
    p.add_argument('--bucket', help='omit for the bucketless transports (inbox and URLs; needs --manifest)')
    p.add_argument('--manifest-key')
    p.add_argument('--manifest')
    p.add_argument('--pointer-key', help='outputs/<job>/<dir>/latest.json: fetch the output manifest it names')
    p.add_argument('--missing-ok', action='store_true', help='a missing pointer is not an error (first launch)')
    p.add_argument('--inbox', type=Path, help='directory of pushed objects named by sha256 (file-API transport)')
    p.add_argument('--root', type=Path, default=Path(os.environ.get('NATLANG_REPO', str(REPO))))
    p.add_argument('--map', action='append', default=[])
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('publish')
    p.add_argument('directory', type=Path)
    p.add_argument('--bucket', required=True)
    p.add_argument('--prefix', required=True, help='outputs/<job id>/ (the pointer is <prefix><dir name>/latest.json)')
    p.add_argument('--note', default='')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('watch', help='publish directories incrementally while a job runs')
    p.add_argument('directories', type=Path, nargs='+')
    p.add_argument('--bucket', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--interval', type=float, default=60)
    p.add_argument('--max-interval', type=float, default=1800)
    p.add_argument('--trigger', action='append', help=f'file-name glob (default {", ".join(DEFAULT_TRIGGERS)})')
    p = sub.add_parser('snapshot', help='write the manifest of a directory (no upload; the file-API transport)')
    p.add_argument('directory', type=Path)
    p.add_argument('-o', '--out', type=Path, required=True)
    p = sub.add_parser('record-exit', help='write <prefix>exit.json with the job status')
    p.add_argument('--bucket', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--status', type=int, required=True)
    p = sub.add_parser('gc', help='list (default) or delete bucket objects no kept root references')
    p.add_argument('--bucket', required=True)
    p.add_argument('--drop-job', action='append', default=[], help='a discarded job: delete its outputs/<job>/')
    p.add_argument('--keep-bundle', action='append', default=[], help='input bundle id to keep regardless of age')
    p.add_argument('--keep-input-days', type=float, default=14)
    p.add_argument('--min-age-hours', type=float, default=24)
    p.add_argument('--out', type=Path, help='write the plan (the exact keys) here')
    p.add_argument('--execute', action='store_true', help='delete exactly the planned keys')
    p = sub.add_parser('ping', help='list a few keys: checks endpoint, region, credentials and bucket policy')
    p.add_argument('--bucket', required=True)
    p.add_argument('--prefix', default='')
    p = sub.add_parser('verify')
    p.add_argument('manifest', type=Path)
    p.add_argument('--root', type=Path, default=REPO)
    p.add_argument('--map', action='append', default=[])
    args = parser.parse_args(argv)
    if args.command == 'fetch' and not args.bucket and not args.manifest:
        parser.error('fetch without --bucket needs --manifest')
    if args.command == 'fetch' and not args.bucket:
        fetch(json.loads(Path(args.manifest).read_text()), None, None, args.root, _mapping(args.map), args.dry_run,
              inbox=args.inbox)
        return 0
    if args.command in ('upload', 'fetch', 'publish', 'ping', 'watch', 'record-exit', 'gc') and not getattr(args, 'dry_run', False):
        _use_client_venv()

    if args.command == 'plan':
        pairs: list[tuple[dict, Path]] = []
        if args.recipe:
            pairs += recipe_entries(args.recipe, args.stage)
        for corpus in args.corpus:
            pairs += corpus_entries(corpus)
        for extra in args.extra:
            pairs += extra_entries(extra)
        for spec in args.hf:
            pairs += hf_entries(spec)
        if args.prebuilt:
            import prebuilt
            pairs += prebuilt.bundle_entries(args.prebuilt, args.code_sha)
        code, code_source = None, None
        staging = Path(tempfile.mkdtemp(prefix='cw-bundle-'))
        if args.code_sha:
            code, code_source = code_archive(args.code_sha, staging)
        manifest = build_manifest(pairs, code, args.note)
        sources = {item['key']: str(source) for item, source in pairs if source is not None}
        if code:
            sources[code['key']] = str(code_source)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=1) + '\n')
        args.out.with_suffix('.sources.json').write_text(json.dumps(sources, indent=1) + '\n')
        print(json.dumps({'bundle': manifest['id'], 'files': len(manifest['files']), 'bytes': manifest['bytes'],
                          'code': code and code['git_sha'], 'manifest': str(args.out)}))
        return 0
    if args.command == 'ping':
        listing = s3_client().list_objects_v2(Bucket=args.bucket, Prefix=args.prefix, MaxKeys=10)
        print(json.dumps({'bucket': args.bucket, 'reachable': True, 'keys': [o['Key'] for o in listing.get('Contents', [])]}))
        return 0
    if args.command == 'verify':
        bad = verify(json.loads(args.manifest.read_text()), args.root, _mapping(args.map))
        print(json.dumps({'verified': not bad, 'mismatched': bad}))
        return 1 if bad else 0
    if args.command == 'upload':
        manifest = json.loads(args.manifest.read_text())
        sources = {k: Path(v) for k, v in json.loads(args.manifest.with_suffix('.sources.json').read_text()).items()}
        upload(manifest, sources, args.bucket, None if args.dry_run else s3_client(), args.dry_run)
        return 0
    if args.command == 'fetch':
        if not (args.manifest or args.manifest_key or args.pointer_key):
            parser.error('fetch needs --manifest-key, --pointer-key or --manifest')
        s3 = None if (args.dry_run and args.manifest) else s3_client()
        manifest = _load_manifest(args, s3)
        if manifest is None:
            print(json.dumps({'event': 'no_pointer', 'pointer': args.pointer_key}))
            return 0 if args.missing_ok else 1
        fetch(manifest, args.bucket, s3, args.root, _mapping(args.map), args.dry_run, inbox=args.inbox)
        return 0
    if args.command == 'publish':
        publish_directory(args.directory, args.bucket, args.prefix, None if args.dry_run else s3_client(),
                          note=args.note, dry_run=args.dry_run)
        return 0
    if args.command == 'gc':
        s3 = s3_client()
        plan = gc_plan(s3, args.bucket, args.drop_job, args.keep_bundle, args.keep_input_days, args.min_age_hours)
        if args.out:
            args.out.write_text(json.dumps(plan, indent=1) + '\n')
        if args.execute:
            plan['deleted'] = gc_execute(s3, args.bucket, plan)
        print(json.dumps({k: v for k, v in plan.items() if k != 'delete'} | {'delete_count': len(plan['delete'])}))
        return 0
    if args.command == 'snapshot':
        manifest, sources = directory_manifest(args.directory, 'snapshot')
        close_sources(sources)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**manifest, 'directory': os.path.abspath(args.directory)}, indent=1) + '\n')
        print(json.dumps({'event': 'snapshot', 'directory': os.path.abspath(args.directory),
                          'files': len(manifest['files']), 'bytes': manifest['bytes']}))
        return 0
    if args.command == 'record-exit':
        record = {'job': os.environ.get('NATLANG_CW_JOB'), 'sandbox': os.environ.get('CW_SANDBOX_ID'),
                  'status': args.status, 'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        s3_client().put_object(Bucket=args.bucket, Key=args.prefix.rstrip('/') + '/exit.json',
                               Body=json.dumps(record).encode())
        print(json.dumps({'event': 'exit_recorded', **record}))
        return 0
    if args.command == 'watch':
        watch(args.directories, args.bucket, args.prefix, s3_client(), args.interval, args.max_interval,
              args.trigger or DEFAULT_TRIGGERS)
        return 0
    return 2


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cwctl import ControllerError, load_env_file  # one env-file reader: ~/.config/natlang/coreweave.env
    try:
        load_env_file()
        sys.exit(main())
    except (BundleError, ControllerError, ValueError) as error:
        print(f'bundle: {error}', file=sys.stderr)
        sys.exit(2)
