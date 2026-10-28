#!/usr/bin/env python3
"""Prebuilt, content-addressed artifacts for CoreWeave sandboxes (registry: scripts/coreweave/prebuilt-artifacts.json).

Sandboxes are x86_64 and often get few CPUs; compiling llama.cpp with CUDA there cost ~45 min (2 CPUs). Artifacts are
built once (x86_64 ones on a CPU sandbox, scripts/coreweave/prebuilt_build.sh; the architecture-independent ts-host
build on DGX), stored by sha256 under the DGX CAS directory, and added to a job's bundle by name:

  bundle.py plan ... --prebuilt llama-cpp-neuralese-sm120 --prebuilt wheelhouse --prebuilt ts-host --code-sha SHA

Each named artifact becomes two bundle files: ``.coreweave-prebuilt/<name>.tar.gz`` (verified like every bundle file)
and ``.coreweave-prebuilt/<name>.json`` (its registry entry). The bootstrap runs ``prebuilt.py install`` after the
fetch, which unpacks them and prints the shell exports it evaluates:

  kind llama-cpp  -> $NATLANG_CW_WORK/prebuilt/<name>/llama-cpp-neuralese/bin; NATLANG_LLAMA_SERVER, NATLANG_LLAMA_BIN_DIR
                     and PATH (ts-host and scripts/harness_bench_matrix.py read NATLANG_LLAMA_SERVER)
  kind wheelhouse -> pip install --no-index --no-deps of the exact wheels; NATLANG_CW_WHEELHOUSE_INSTALLED=1 makes the
                     bootstrap skip its networked requirements install
  kind ts-host    -> ts-host/dist and ts-host/node_modules in the repo; NATLANG_CW_TSHOST_PREBUILT=1 skips npm ci/build

Bindings that keep a reused artifact honest:
  - image-bound kinds (llama-cpp, wheelhouse) record the image digest they were built in; cwctl.launch refuses a job
    whose image differs (CUDA runtime/cuBLAS and the Python ABI come from the image).
  - ts-host records the git tree ids of its build inputs (TSHOST_INPUTS); ``plan`` refuses it for a code commit whose
    trees differ, so a stale dist never ships (rebuild with ``prebuilt.py build-tshost --code-sha SHA``).

  prebuilt.py register NAME --kind K --tar FILE [--image IMG] [--meta JSON]   copy into the CAS, update the registry
  prebuilt.py build-tshost --code-sha SHA [--name ts-host]                     build + register (run under the ledger)
  prebuilt.py install --root REPO --work WORK                                  (in the sandbox; stdlib only)
  prebuilt.py show
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REGISTRY = HERE / 'prebuilt-artifacts.json'
SCHEMA = 'natlang.coreweave-prebuilt/1'
CAS = Path(os.environ.get('NATLANG_CW_PREBUILT_CAS', '/home/werg/data/coreweave-prebuilt/cas'))
BUNDLE_DIR = '.coreweave-prebuilt'
KINDS = ('llama-cpp', 'wheelhouse', 'ts-host')
IMAGE_BOUND = ('llama-cpp', 'wheelhouse')
# Everything the ts-host build reads (package scripts: generate-builtin, bundle-improver, stage-npm-package).
TSHOST_INPUTS = ('ts-host', 'applications/program-improver', 'scripts/stage-npm-package.mjs', 'examples/adaptation')
NODE_DIR = Path(os.environ.get('NATLANG_NODE_DIR', '/home/werg/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin'))


class PrebuiltError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path = REGISTRY) -> dict:
    if not path.is_file():
        return {'schema': SCHEMA, 'cas': str(CAS), 'artifacts': {}}
    registry = json.loads(path.read_text())
    if registry.get('schema') != SCHEMA:
        raise PrebuiltError(f'{path}: not a {SCHEMA} registry')
    return registry


def cas_path(sha256: str) -> Path:
    return CAS / sha256[:2] / f'{sha256}.tar.gz'


def git_trees(code_sha: str, repo: Path = REPO) -> dict[str, str]:
    """Git object ids of the ts-host build inputs at a commit (blob or tree id; '' when absent)."""
    out = {}
    for path in TSHOST_INPUTS:
        result = subprocess.run(['git', '-C', str(repo), 'rev-parse', f'{code_sha}:{path}'], capture_output=True,
                                text=True)
        out[path] = result.stdout.strip() if result.returncode == 0 else ''
    return out


def register(name: str, kind: str, tar: Path, image: str | None = None, meta: dict | None = None,
             registry_path: Path = REGISTRY) -> dict:
    if kind not in KINDS:
        raise PrebuiltError(f'kind is one of {", ".join(KINDS)}')
    if kind in IMAGE_BOUND and not image:
        raise PrebuiltError(f'a {kind} artifact is bound to the image it was built in: pass --image')
    digest = sha256_file(tar)
    target = cas_path(digest)
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f'.{target.name}.partial')
        shutil.copyfile(tar, temp)
        if sha256_file(temp) != digest:
            temp.unlink()
            raise PrebuiltError(f'{tar}: copy into the CAS changed its bytes')
        temp.replace(target)
    entry = {'kind': kind, 'sha256': digest, 'bytes': target.stat().st_size, 'image': image,
             'registered': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **(meta or {})}
    registry = load_registry(registry_path)
    registry['artifacts'][name] = entry
    registry_path.write_text(json.dumps(registry, indent=1, sort_keys=False) + '\n')
    return {'name': name, **entry, 'path': str(target)}


def bundle_entries(names: list[str], code_sha: str | None, registry_path: Path = REGISTRY,
                   repo: Path = REPO) -> list[tuple[dict, Path]]:
    """(entry, source) pairs for bundle.py plan: the artifact and its descriptor, both content-addressed."""
    from bundle import entry  # bundle.py's own entry builder (sha256, bytes, CAS key)
    registry = load_registry(registry_path)
    out = []
    for name in names:
        item = registry['artifacts'].get(name)
        if item is None:
            raise PrebuiltError(f'no prebuilt artifact {name!r}; registered: {", ".join(sorted(registry["artifacts"]))}')
        if item['kind'] == 'ts-host':
            if not code_sha:
                raise PrebuiltError('ts-host is bound to its source trees: plan it with --code-sha')
            actual = git_trees(code_sha, repo)
            if actual != item.get('inputs'):
                changed = sorted(p for p in TSHOST_INPUTS if actual.get(p) != (item.get('inputs') or {}).get(p))
                raise PrebuiltError(f'ts-host artifact was built from other sources ({", ".join(changed)} differ at '
                                    f'{code_sha}); rebuild: prebuilt.py build-tshost --code-sha {code_sha}')
        source = cas_path(item['sha256'])
        if not source.is_file():
            raise PrebuiltError(f'{name}: CAS object {source} is missing')
        descriptor = CAS / 'descriptors' / f'{name}-{item["sha256"][:16]}.json'
        text = json.dumps({'name': name, **item}, indent=1, sort_keys=True) + '\n'
        if not descriptor.is_file() or descriptor.read_text() != text:
            descriptor.parent.mkdir(parents=True, exist_ok=True)
            descriptor.write_text(text)
        out.append((entry(f'{BUNDLE_DIR}/{name}.tar.gz', source, item['sha256']), source))
        out.append((entry(f'{BUNDLE_DIR}/{name}.json', descriptor), descriptor))
    return out


def check_job(job: dict, manifest: dict, sources: dict[str, str]) -> list[str]:
    """Names of the bundle's prebuilt artifacts; raises when an image-bound one was built in another image."""
    names = []
    for item in manifest.get('files', []):
        path = item['path']
        if not (path.startswith(f'{BUNDLE_DIR}/') and path.endswith('.json')):
            continue
        descriptor = json.loads(Path(sources[item['key']]).read_text())
        names.append(descriptor['name'])
        if descriptor['kind'] in IMAGE_BOUND and descriptor.get('image') != job['image']:
            raise PrebuiltError(f'prebuilt {descriptor["name"]} was built in {descriptor.get("image")}, but job '
                                f'{job["id"]} runs {job["image"]}: use that image or rebuild the artifact')
    return names


def build_tshost(code_sha: str, name: str = 'ts-host', repo: Path = REPO) -> dict:
    """npm ci + build of ts-host at a commit (git archive, so the worktree's state never leaks in), then prune to
    the production dependencies and pack ts-host/dist + ts-host/node_modules. JavaScript only: built on DGX."""
    resolved = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--verify', f'{code_sha}^{{commit}}'],
                              capture_output=True, text=True, check=True).stdout.strip()
    env = {**os.environ, 'PATH': f'{NODE_DIR}:{os.environ.get("PATH", "")}'}
    with tempfile.TemporaryDirectory(prefix='prebuilt-tshost-') as temp:
        root = Path(temp) / 'src'
        root.mkdir()
        archive = subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', resolved], capture_output=True,
                                 check=True).stdout
        subprocess.run(['tar', '-x', '-C', str(root)], input=archive, check=True)
        started = time.time()
        for command in (['npm', 'ci', '--no-audit', '--no-fund'], ['npm', 'run', 'build'],
                        ['npm', 'prune', '--omit=dev', '--no-audit', '--no-fund']):
            subprocess.run(command, cwd=root / 'ts-host', env=env, check=True, stdout=sys.stderr)
        seconds = round(time.time() - started, 1)
        native = sorted(str(p.relative_to(root)) for p in (root / 'ts-host' / 'node_modules').rglob('*.node'))
        if native:
            raise PrebuiltError(f'native addons would make the ts-host artifact architecture-bound: {native[:5]}')
        tar = Path(temp) / f'{name}.tar.gz'
        subprocess.run(f'tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C {root} -cf - '
                       'ts-host/dist ts-host/node_modules | gzip -n -9 > ' + str(tar), shell=True, check=True)
        node = subprocess.run([str(NODE_DIR / 'node'), '--version'], capture_output=True, text=True).stdout.strip()
        return register(name, 'ts-host', tar, None, {'git_sha': resolved, 'inputs': git_trees(resolved, repo),
                                                     'node': node, 'build_seconds': seconds,
                                                     'architecture_independent': True})


def install(root: Path, work: Path, log=lambda text: print(text, file=sys.stderr)) -> list[str]:
    """In the sandbox, after bundle fetch: unpack every fetched artifact; return shell export lines."""
    exports: list[str] = []
    directory = root / BUNDLE_DIR
    for descriptor_path in sorted(directory.glob('*.json')) if directory.is_dir() else []:
        descriptor = json.loads(descriptor_path.read_text())
        name, kind = descriptor['name'], descriptor['kind']
        tar_path = directory / f'{name}.tar.gz'
        if sha256_file(tar_path) != descriptor['sha256']:
            raise PrebuiltError(f'{tar_path}: sha256 differs from its descriptor')
        started = time.time()
        if kind == 'ts-host':
            with tarfile.open(tar_path) as tar:
                tar.extractall(root, filter='data')  # relative file: links of vendored packages stay inside the repo
            exports.append('export NATLANG_CW_TSHOST_PREBUILT=1')
        else:
            target = work / 'prebuilt' / name
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True)
            with tarfile.open(tar_path) as tar:
                tar.extractall(target, filter='data')
            if kind == 'llama-cpp':
                bin_dir = target / 'llama-cpp-neuralese' / 'bin'
                exports += [f'export NATLANG_LLAMA_BIN_DIR={bin_dir}',
                            f'export NATLANG_LLAMA_SERVER={bin_dir}/llama-server',
                            f'export PATH={bin_dir}:$PATH']
            elif kind == 'wheelhouse':
                wheels = target / 'wheelhouse'
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                                '--find-links', str(wheels), '-r', str(wheels / 'added.txt')], check=True,
                               stdout=sys.stderr)
                exports.append('export NATLANG_CW_WHEELHOUSE_INSTALLED=1')
        log(json.dumps({'bootstrap': 'prebuilt', 'name': name, 'kind': kind, 'sha256': descriptor['sha256'],
                        'seconds': round(time.time() - started, 1)}))
    return exports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('register')
    p.add_argument('name')
    p.add_argument('--kind', required=True, choices=KINDS)
    p.add_argument('--tar', type=Path, required=True)
    p.add_argument('--image')
    p.add_argument('--meta', help='JSON object of provenance fields')
    p = sub.add_parser('build-tshost')
    p.add_argument('--code-sha', required=True)
    p.add_argument('--name', default='ts-host')
    p = sub.add_parser('install')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    sub.add_parser('show')
    args = parser.parse_args(argv)
    if args.command == 'register':
        print(json.dumps(register(args.name, args.kind, args.tar, args.image,
                                  json.loads(args.meta) if args.meta else None), indent=1))
    elif args.command == 'build-tshost':
        print(json.dumps(build_tshost(args.code_sha, args.name), indent=1))
    elif args.command == 'install':
        print('\n'.join(install(args.root, args.work)))
    else:
        print(json.dumps(load_registry(), indent=1))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except PrebuiltError as error:
        print(f'prebuilt: {error}', file=sys.stderr)
        sys.exit(2)
