#!/usr/bin/env python3
"""Freeze, verify, and optionally launch the free typed-decision pool.

The default operation preserves the old ``freeze_decision_pool.py DEST`` CLI.
For launches use ``run BUNDLE -- RUNNER_ARGS``; this verifies every frozen byte
and forces the runner to consume the bundle's pinned models.json before exec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'scripts/run_gemini_decision_pool.py',
    'scripts/label_decision_cases.py',
    'training/neuralese/natlang_neuralese/__init__.py',
    'training/neuralese/natlang_neuralese/common/__init__.py',
    'training/neuralese/natlang_neuralese/common/jsonio.py',
)
POOL_SCHEMA = 'natlang.free-text-model-pool/1'
FREEZE_SCHEMA = 'natlang.frozen-decision-pool/1'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_model_pool(path: Path) -> dict:
    try:
        pool = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'{path}: cannot read model pool: {exc}') from exc
    if not isinstance(pool, dict) or pool.get('schema') != POOL_SCHEMA:
        raise ValueError(f'{path}: expected schema {POOL_SCHEMA}')
    models = pool.get('models')
    if not isinstance(models, list) or not models:
        raise ValueError(f'{path}: models must be a nonempty array')
    ids = set()
    for index, model in enumerate(models):
        where = f'{path}:models[{index}]'
        if not isinstance(model, dict):
            raise ValueError(f'{where}: expected an object')
        model_id, quota_group = model.get('id'), model.get('quota_group')
        if not isinstance(model_id, str) or not model_id or model_id in ids:
            raise ValueError(f'{where}: id must be a unique nonempty string')
        ids.add(model_id)
        if not isinstance(quota_group, str) or not quota_group:
            raise ValueError(f'{where}: quota_group must be a nonempty string')
        if model.get('free_text_verified') is not True:
            raise ValueError(f'{where}: free_text_verified must be true')
        effort = model.get('reasoning_effort')
        if effort not in ('omit', 'minimal', 'low', 'medium', 'high'):
            raise ValueError(f'{where}: unsupported reasoning_effort {effort!r}')
        interval = model.get('interval_seconds')
        if (isinstance(interval, bool) or not isinstance(interval, (int, float)) or
                not math.isfinite(interval) or interval <= 0):
            raise ValueError(f'{where}: interval_seconds must be finite and positive')
    return pool


def _safe_relative(raw: object, where: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f'{where}: path must be a nonempty string')
    posix = PurePosixPath(raw)
    if posix.is_absolute() or any(part in ('', '.', '..') for part in posix.parts):
        raise ValueError(f'{where}: path must be normalized and relative: {raw!r}')
    return Path(*posix.parts)


def verify_bundle(bundle: Path) -> dict:
    bundle = bundle.resolve(strict=True)
    manifest_path = bundle / 'frozen-runtime.json'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'{manifest_path}: cannot read frozen manifest: {exc}') from exc
    if not isinstance(manifest, dict) or manifest.get('schema') != FREEZE_SCHEMA:
        raise ValueError(f'{manifest_path}: expected schema {FREEZE_SCHEMA}')
    if manifest.get('training_admission') is not False:
        raise ValueError(f'{manifest_path}: training_admission must be false')
    entries = manifest.get('files')
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'{manifest_path}: files must be a nonempty array')
    declared: set[str] = set()
    verified = []
    for index, entry in enumerate(entries):
        where = f'{manifest_path}:files[{index}]'
        if not isinstance(entry, dict):
            raise ValueError(f'{where}: expected an object')
        rel = _safe_relative(entry.get('path'), where)
        rel_str = rel.as_posix()
        if rel_str in declared:
            raise ValueError(f'{where}: duplicate file path {rel_str}')
        declared.add(rel_str)
        target = bundle / rel
        if target.is_symlink() or not target.is_file():
            raise ValueError(f'{where}: listed path must be an in-bundle regular file: {rel_str}')
        resolved = target.resolve(strict=True)
        if not resolved.is_relative_to(bundle):
            raise ValueError(f'{where}: listed path escapes bundle: {rel_str}')
        data = target.read_bytes()
        actual_sha = digest(data)
        if (isinstance(entry.get('bytes'), bool) or entry.get('bytes') != len(data) or
                entry.get('sha256') != actual_sha):
            raise ValueError(f'{where}: byte count or SHA-256 mismatch for {rel_str}')
        verified.append({'path': rel_str, 'bytes': len(data), 'sha256': actual_sha})

    actual_files = set()
    ignored_bytecode_caches = []
    for path in bundle.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'{bundle}: symlinks are not allowed in a frozen pool bundle: {path.relative_to(bundle)}')
        if path.is_file() and path != manifest_path:
            relative = path.relative_to(bundle).as_posix()
            if path.parent.name == '__pycache__' and path.suffix == '.pyc':
                data = path.read_bytes()
                ignored_bytecode_caches.append({'path': relative, 'bytes': len(data), 'sha256': digest(data)})
            else:
                actual_files.add(relative)
    if actual_files != declared:
        missing = sorted(declared - actual_files)
        unlisted = sorted(actual_files - declared)
        raise ValueError(f'{bundle}: manifest file set differs from bundle; missing={missing}, unlisted={unlisted}')

    entrypoint = _safe_relative(manifest.get('entrypoint'), f'{manifest_path}:entrypoint')
    if entrypoint.as_posix() not in declared:
        raise ValueError(f'{manifest_path}: entrypoint is not listed in frozen files')
    required = set(FILES)
    if not required.issubset(declared) or 'models.json' not in declared:
        raise ValueError(f'{manifest_path}: bundle omits a required runner/import/model file')
    model_path = bundle / 'models.json'
    pool = validate_model_pool(model_path)
    return {'bundle': str(bundle), 'manifest_sha256': digest(manifest_path.read_bytes()),
            'files_verified': len(verified), 'models_verified': len(pool['models']),
            'entrypoint': entrypoint.as_posix(), 'verified_files': verified,
            'ignored_python_bytecode_caches': ignored_bytecode_caches}


def freeze(destination: Path, models_path: Path) -> dict:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f'output directory must be fresh: {destination}')
    pool = validate_model_pool(models_path.resolve(strict=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Build in a sibling staging directory; publish only after all copied bytes
    # have been checked against the just-written manifest.
    staging = destination.with_name(destination.name + '.freeze-staging')
    if staging.exists():
        raise FileExistsError(f'staging directory already exists: {staging}')
    staging.mkdir()
    try:
        entries = []
        sources = [(ROOT / name, name) for name in FILES]
        sources.append((models_path.resolve(), 'models.json'))
        for source, relative in sources:
            if not source.is_file() or source.is_symlink():
                raise ValueError(f'{source}: expected a regular source file')
            data = source.read_bytes()
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            entries.append({'path': relative, 'bytes': len(data), 'sha256': digest(data)})
        manifest = {'schema': FREEZE_SCHEMA,
                    'entrypoint': 'scripts/run_gemini_decision_pool.py',
                    'model_pool_schema': POOL_SCHEMA,
                    'files': entries, 'training_admission': False}
        (staging / 'frozen-runtime.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        verify_bundle(staging)
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_bundle(destination)


def _parse_args(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward-compatible shorthand: DEST [--models PATH] still means freeze.
    if argv and argv[0] not in {'freeze', 'verify', 'run', '-h', '--help'}:
        argv.insert(0, 'freeze')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    freezing = sub.add_parser('freeze', help='copy the live local pool code and models into a fresh bundle')
    freezing.add_argument('destination', type=Path)
    freezing.add_argument('--models', type=Path,
                          default=ROOT / 'training/providers/gemini-free-text-pool.json')
    verify = sub.add_parser('verify', help='verify every listed byte and the pool schema before launch')
    verify.add_argument('bundle', type=Path)
    run = sub.add_parser('run', help='verify a frozen bundle, then exec its runner')
    run.add_argument('bundle', type=Path)
    run.add_argument('runner_args', nargs=argparse.REMAINDER,
                     help='arguments for the pinned runner, after --')
    return parser.parse_args(argv)


def _model_arg(args: list[str], bundle_models: Path) -> list[str]:
    supplied_values = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == '--models':
            if index + 1 >= len(args):
                raise ValueError('--models requires a path')
            supplied_values.append(args[index + 1])
            index += 2
            continue
        if arg.startswith('--models='):
            supplied_values.append(arg.partition('=')[2])
        index += 1
    if len(supplied_values) > 1:
        raise ValueError('runner --models may be supplied at most once')
    if not supplied_values:
        return [*args, '--models', str(bundle_models)]
    supplied = supplied_values[0]
    if Path(supplied).resolve() != bundle_models.resolve():
        raise ValueError(f'runner --models must point to verified bundle file {bundle_models}')
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.operation == 'freeze':
        result = freeze(args.destination, args.models)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.operation == 'verify':
        result = verify_bundle(args.bundle)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    result = verify_bundle(args.bundle)
    bundle = Path(result['bundle'])
    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == '--':
        runner_args.pop(0)
    if not runner_args:
        raise ValueError('run requires arguments for --cases, --out, and --state')
    runner_args = _model_arg(runner_args, bundle / 'models.json')
    entrypoint = bundle / result['entrypoint']
    environment = os.environ.copy()
    # Keep Python from reading or adding unmanifested .pyc files in the bundle.
    # A fresh external cache prefix prevents existing bundle caches from being
    # used even though ordinary cache files are excluded from source identity.
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONPYCACHEPREFIX'] = tempfile.mkdtemp(prefix='natlang-google-pool-pycache-')
    os.execvpe(sys.executable, [sys.executable, str(entrypoint), *runner_args], environment)
    return 127


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'freeze_decision_pool: {exc}', file=sys.stderr)
        raise SystemExit(2)
