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

  plan     --recipe R [--stage S ...] [--corpus ID ...] [--extra PATH ...] [--code-sha SHA] -o manifest.json
  upload   manifest.json --bucket B [--dry-run]
  fetch    --bucket B (--manifest-key K | --pointer-key outputs/<job>/<dir>/latest.json | --manifest FILE) [--map OLD=NEW ...] [--dry-run]
  publish  DIR --bucket B --prefix outputs/<job>/ [--dry-run]
  verify   manifest.json [--map OLD=NEW ...]
  ping     --bucket B [--prefix P]

Credentials: never in files or arguments. Inside a sandbox the AWS SDK reads the sandbox's injected Workload Identity
Federation token (AWS_CONTAINER_CREDENTIALS_FULL_URI etc.); on DGX/Pop the CoreWeave API token from
COREWEAVE_API_TOKEN or CWSANDBOX_API_KEY is exchanged for temporary credentials through the documented
api-token endpoint. Requires boto3 >= 1.42.5 (older versions reject the non-loopback credentials host).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex_cached  # noqa: E402

SCHEMA = 'natlang.coreweave-bundle/1'
DEFAULT_ENDPOINT = 'https://cwobject.com'
TOKEN_EXCHANGE_URI = 'https://api.coreweave.com/v1/cwobject/temporary-credentials/api-token'


class BundleError(ValueError):
    pass


def object_key(sha256: str) -> str:
    if len(sha256) != 64 or any(c not in '0123456789abcdef' for c in sha256):
        raise BundleError(f'not a lowercase sha256 hex digest: {sha256!r}')
    return f'cas/sha256/{sha256[:2]}/{sha256}'


def manifest_id(manifest: dict[str, Any]) -> str:
    return canonical_json_sha256_hex({k: v for k, v in manifest.items() if k not in ('id', 'created')})


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
    return boto3.client('s3', endpoint_url=os.environ.get('AWS_ENDPOINT_URL_S3', DEFAULT_ENDPOINT),
                        config=Config(region_name=region, s3={'addressing_style': 'virtual'}))


def _exists(s3, bucket: str, key: str, size: int) -> bool:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # botocore ClientError 404 / NoSuchKey
        if getattr(error, 'response', {}).get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
            return False
        raise
    return head.get('ContentLength') == size


def upload(manifest: dict[str, Any], sources: dict[str, Path], bucket: str, s3, dry_run: bool = False,
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
        log(json.dumps({'event': 'upload', 'key': item['key'], 'bytes': item['bytes'], 'from': str(source),
                        'dry_run': dry_run}))
        if not dry_run:
            s3.upload_file(str(source), bucket, item['key'])
        stats['uploaded'] += 1
        stats['bytes_uploaded'] += item['bytes']
    if not dry_run:
        s3.put_object(Bucket=bucket, Key=manifest_key(manifest), Body=json.dumps(manifest, indent=1).encode())
    log(json.dumps({'event': 'manifest', 'key': manifest_key(manifest), 'dry_run': dry_run, **stats}))
    return stats


def fetch(manifest: dict[str, Any], bucket: str, s3, root: Path, mapping: list[tuple[str, str]] = (),
          dry_run: bool = False, log=print) -> dict[str, Any]:
    """Materialize every file, verifying each sha256 before it is moved into place; write a receipt."""
    check_manifest(manifest)
    mapping = list(mapping)
    fetched, present = 0, 0
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
            s3.download_file(bucket, item['key'], str(temp))
            actual = sha256_file_hex(temp)
            if actual != item['sha256']:
                raise BundleError(f'{item["path"]}: downloaded sha256 {actual} != manifest {item["sha256"]}')
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        fetched += 1
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


def directory_manifest(directory: Path, note: str = '') -> tuple[dict, dict[str, Path]]:
    """Manifest of a directory tree under its absolute path (a stage --out), for publishing outputs."""
    directory = directory.resolve()
    pairs = []
    for child in sorted(p for p in directory.rglob('*') if p.is_file() and not p.name.startswith('.cw-fetch-')):
        if child.name.endswith('.reserve'):  # preallocated free-space reservation, not content
            continue
        pairs.append((entry(str(child), child, sha256_file_hex(child)), child))
    manifest = build_manifest(pairs, note=note)
    return manifest, {item['key']: source for item, source in pairs}


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
        pointer = json.loads(s3.get_object(Bucket=args.bucket, Key=args.pointer_key)['Body'].read())
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
    p.add_argument('--code-sha', help='commit to package as the code snapshot (git archive)')
    p.add_argument('--note', default='')
    p.add_argument('-o', '--out', type=Path, required=True)
    p = sub.add_parser('upload')
    p.add_argument('manifest', type=Path)
    p.add_argument('--bucket', required=True)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('fetch')
    p.add_argument('--bucket', required=True)
    p.add_argument('--manifest-key')
    p.add_argument('--manifest')
    p.add_argument('--pointer-key', help='outputs/<job>/<dir>/latest.json: fetch the output manifest it names')
    p.add_argument('--root', type=Path, default=Path(os.environ.get('NATLANG_REPO', str(REPO))))
    p.add_argument('--map', action='append', default=[])
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('publish')
    p.add_argument('directory', type=Path)
    p.add_argument('--bucket', required=True)
    p.add_argument('--prefix', required=True, help='e.g. outputs/<job id>/')
    p.add_argument('--note', default='')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('ping', help='list a few keys: checks endpoint, region, credentials and bucket policy')
    p.add_argument('--bucket', required=True)
    p.add_argument('--prefix', default='')
    p = sub.add_parser('verify')
    p.add_argument('manifest', type=Path)
    p.add_argument('--root', type=Path, default=REPO)
    p.add_argument('--map', action='append', default=[])
    args = parser.parse_args(argv)
    if args.command in ('upload', 'fetch', 'publish', 'ping') and not getattr(args, 'dry_run', False):
        _use_client_venv()

    if args.command == 'plan':
        pairs: list[tuple[dict, Path]] = []
        if args.recipe:
            pairs += recipe_entries(args.recipe, args.stage)
        for corpus in args.corpus:
            pairs += corpus_entries(corpus)
        for extra in args.extra:
            pairs += extra_entries(extra)
        code, code_source = None, None
        staging = Path(tempfile.mkdtemp(prefix='cw-bundle-'))
        if args.code_sha:
            code, code_source = code_archive(args.code_sha, staging)
        manifest = build_manifest(pairs, code, args.note)
        sources = {item['key']: str(source) for item, source in pairs}
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
        fetch(_load_manifest(args, s3), args.bucket, s3, args.root, _mapping(args.map), args.dry_run)
        return 0
    if args.command == 'publish':
        manifest, sources = directory_manifest(args.directory, args.note)
        s3 = None if args.dry_run else s3_client()
        upload(manifest, sources, args.bucket, s3, args.dry_run)
        pointer = {'bundle': manifest['id'], 'manifest_key': manifest_key(manifest), 'directory': str(args.directory),
                   'files': len(manifest['files']), 'bytes': manifest['bytes'], 'at': manifest['created']}
        if not args.dry_run:
            s3.put_object(Bucket=args.bucket, Key=args.prefix.rstrip('/') + '/latest.json',
                          Body=json.dumps(pointer, indent=1).encode())
        print(json.dumps({'event': 'published', **pointer, 'dry_run': args.dry_run}))
        return 0
    return 2


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cwctl import ControllerError, load_env_file  # one env-file reader: ~/.config/natlang/coreweave.env
    try:
        load_env_file()
        sys.exit(main())
    except (BundleError, ControllerError) as error:
        print(f'bundle: {error}', file=sys.stderr)
        sys.exit(2)
