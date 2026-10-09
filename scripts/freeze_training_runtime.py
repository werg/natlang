#!/usr/bin/env python3
"""Freeze built Node runtime/scripts so concurrent development cannot change replay."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import tempfile
import sys
import subprocess
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_pipeline import atomic_json, digest_file


def tree_identity(root, compiled_dist=None, compiled_src=None):
    folders = {name: root / name for name in ("dist", "scripts", "src")}
    if compiled_dist is not None:
        folders["dist"] = compiled_dist
    if compiled_src is not None:
        folders["src"] = compiled_src
    return {str(Path(name) / path.relative_to(folder)): digest_file(path)
            for name, folder in folders.items() for path in sorted(folder.rglob("*")) if path.is_file()} | {
                name: digest_file(root / name) for name in ("prelude.js", "package.json", "package-lock.json") if (root / name).exists()}


def check_prompt_features(root, compiled_dist=None):
    """Refuse a new snapshot whose prompt advertises an absent eval feature."""
    native = (compiled_dist if compiled_dist is not None else root / 'dist') / 'native'
    prompt = (native / 'prompt.js').read_text()
    if re.search(r'\bfinish\s*:\s*true\b', prompt):
        agent = (native / 'agent.js').read_text()
        runtime = (native / 'runtime.js').read_text()
        if (not re.search(r"\bfinish\s*:\s*\{\s*type\s*:\s*['\"]boolean['\"]", agent)
                or not re.search(r'\bargs\.finish\s*===\s*true\b', runtime)):
            raise ValueError('Prompt advertises eval finish:true without its compiled tool schema and runtime implementation')


def _build_input_identity(source):
    """Hash the files that determine an isolated Node compile, excluding generated bundle output."""
    repo = source.parent
    files = {}
    for base in (source / 'src', repo / 'applications' / 'program-improver'):
        if not base.exists():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file():
                continue
            if path == source / 'src' / 'improvement' / 'authored-source.ts':
                continue
            files[str(path.relative_to(repo))] = digest_file(path)
    bundle_script = source / 'scripts' / 'bundle-improver.mjs'
    if bundle_script.is_file():
        files[str(bundle_script.relative_to(repo))] = digest_file(bundle_script)
    for name in ('tsconfig.json', 'package.json', 'package-lock.json', 'prelude.js'):
        path = source / name
        if path.is_file():
            files[str(path.relative_to(repo))] = digest_file(path)
    for name in ('node_modules/typescript/package.json', 'node_modules/typescript/bin/tsc',
                 'node_modules/typescript/lib/tsc.js', 'node_modules/typescript/lib/_tsc.js',
                 'node_modules/typescript/lib/typescript.js'):
        path = source / name
        if path.is_file():
            files[str(path.relative_to(repo))] = digest_file(path)
    return files


@contextmanager
def isolated_compiled_runtime(source, work_parent):
    """Compile current Node TypeScript in a disposable source copy; never write canonical dist/src."""
    source = Path(source).resolve()
    repo = source.parent
    apps = repo / 'applications' / 'program-improver'
    bundle_script = source / 'scripts' / 'bundle-improver.mjs'
    tsc = source / 'node_modules' / 'typescript' / 'bin' / 'tsc'
    if not bundle_script.is_file() or not tsc.is_file() or not apps.is_dir():
        raise ValueError('isolated Node build inputs are incomplete (bundle script, TypeScript, or improver sources missing)')
    before = _build_input_identity(source)
    with tempfile.TemporaryDirectory(prefix='runtime-dist-build-', dir=work_parent) as temp:
        build_root = Path(temp)
        build_host = build_root / 'ts-host'
        build_host.mkdir()
        shutil.copytree(source / 'src', build_host / 'src')
        (build_host / 'src' / 'improvement').mkdir(parents=True, exist_ok=True)
        (build_host / 'scripts').mkdir()
        shutil.copy2(bundle_script, build_host / 'scripts' / 'bundle-improver.mjs')
        for name in ('tsconfig.json', 'package.json', 'package-lock.json', 'prelude.js'):
            if (source / name).is_file():
                shutil.copy2(source / name, build_host / name)
        build_host.joinpath('node_modules').symlink_to(source / 'node_modules', target_is_directory=True)
        # The canonical checkout also keeps workspace dependencies at its root;
        # preserve Node/TypeScript ancestor lookup without copying or installing them.
        if (repo / 'node_modules').is_dir():
            build_root.joinpath('node_modules').symlink_to(repo / 'node_modules', target_is_directory=True)
        build_apps = build_root / 'applications' / 'program-improver'
        shutil.copytree(apps, build_apps)
        generated = subprocess.run(['node', str(build_host / 'scripts' / 'bundle-improver.mjs')],
                                   cwd=build_host, text=True, capture_output=True)
        if generated.returncode:
            raise RuntimeError(f'isolated bundle generation failed: {generated.stderr[-3000:]}')
        dist = build_host / 'dist'
        compiled = subprocess.run(['node', str(tsc), '-p', str(build_host / 'tsconfig.json'), '--outDir', str(dist)],
                                  cwd=build_host, text=True, capture_output=True)
        if compiled.returncode:
            details = (compiled.stderr + compiled.stdout)[-5000:]
            raise RuntimeError(f'isolated TypeScript compilation failed (exit {compiled.returncode}): {details}')
        if not (dist / 'native' / 'runtime.js').is_file():
            raise ValueError('isolated TypeScript compilation omitted dist/native/runtime.js')
        if before != _build_input_identity(source):
            raise ValueError('runtime source inputs changed during isolated compilation; retry from a stable checkout')
        compiler_package = json.loads((source / 'node_modules/typescript/package.json').read_text())
        yield {'dist': dist, 'src': build_host / 'src',
               'build': {'mode': 'isolated_typescript',
                         'source_inputs': before,
                         'source_inputs_sha256': hashlib.sha256(json.dumps(before, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                         'compiler': 'ts-host/node_modules/typescript/bin/tsc',
                         'compiler_version': compiler_package.get('version'),
                         'bundle_generator': 'ts-host/scripts/bundle-improver.mjs'}}


def _freeze_new(source, output, compiled_dist, compiled_src=None, build=None):
    # World bridges resolve vendor beside ts-host. Preserve that layout for frozen copies.
    source_vendor, frozen_vendor = source.parent / 'vendor', output.parent / 'vendor'
    if source_vendor.exists() and not frozen_vendor.exists():
        frozen_vendor.parent.mkdir(parents=True, exist_ok=True)
        frozen_vendor.symlink_to(source_vendor, target_is_directory=True)
    if source_vendor.exists() and frozen_vendor.resolve() != source_vendor.resolve():
        raise ValueError('frozen runtime vendor path points at a different dependency tree')
    if build is not None and build.get('mode') == 'isolated_typescript':
        if _build_input_identity(source) != build['source_inputs']:
            raise ValueError('runtime source inputs changed after isolated compilation; retry from a stable checkout')
    before = tree_identity(source, compiled_dist, compiled_src)
    if not before or not (compiled_dist / "native/runtime.js").exists():
        raise ValueError("build the Node runtime before freezing it")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"freeze destination appeared during build: {output}")
    check_prompt_features(source, compiled_dist)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runtime-building-", dir=output.parent) as temp:
        staging = Path(temp) / 'runtime'
        staging.mkdir()
        for name in ("dist", "scripts", "src"):
            selected = compiled_dist if name == 'dist' else compiled_src if name == 'src' and compiled_src is not None else source / name
            shutil.copytree(selected, staging / name)
        for name in ("prelude.js", "package.json", "package-lock.json"):
            if (source / name).exists():
                shutil.copy2(source / name, staging / name)
        (staging / "node_modules").symlink_to(source / "node_modules", target_is_directory=True)
        copied = tree_identity(staging)
        if before != copied or before != tree_identity(source, compiled_dist, compiled_src):
            raise ValueError("runtime changed while freezing; retry after the build finishes")
        if build is not None and build.get('mode') == 'isolated_typescript' and _build_input_identity(source) != build['source_inputs']:
            raise ValueError('runtime source inputs changed while freezing; retry from a stable checkout')
        data = {"version": "natlang.frozen_runtime/1", "files": copied,
                "compiled_dist_source": str(compiled_dist) if build is None else 'isolated build created for this freeze',
                "build": build or {'mode': 'explicit_compiled_dist', 'compiled_dist_source': str(compiled_dist),
                                    'compiled_dist_files': {key.removeprefix('dist/'): value
                                                            for key, value in tree_identity(source, compiled_dist).items()
                                                            if key.startswith('dist/')}},
                "node_modules": str(source / "node_modules"),
                "note": "Runtime and scripts copied; installed Node dependencies are shared and must not be changed during a run."}
        atomic_json(staging / "frozen-runtime.json", data)
        staging.rename(output)
        return data


def freeze(source, output, compiled_dist=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = output / "frozen-runtime.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if tree_identity(output) != data["files"]:
            raise ValueError("frozen runtime changed")
        # An existing snapshot is immutable and reusable without compiling newer source.
        return data
    # A caller-supplied dist is an explicit reviewed input. The default always builds
    # current source in isolation, so a stale canonical dist cannot hide source edits.
    if compiled_dist is not None:
        return _freeze_new(source, output, Path(compiled_dist).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with isolated_compiled_runtime(source, output.parent) as built:
        return _freeze_new(source, output, built['dist'], built['src'], built['build'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compiled-dist", type=Path,
                        help="Use this explicit reviewed precompiled dist; default compiles current TypeScript in isolation")
    args = parser.parse_args()
    print(json.dumps({"files": len(freeze(args.source, args.output, args.compiled_dist)["files"]), "output": str(args.output)}))
