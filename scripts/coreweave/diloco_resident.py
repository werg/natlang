#!/usr/bin/env python3
"""Stage and publish a native resident source revision to every DiLoCo worker.

publish-source-update RUN_RECORD MANIFEST --journal PATH stages immutable byte-verified files,
prepares every worker against its own live predecessor and geometry, then publishes CAS pointers.
status RUN_RECORD reads actual per-worker resident commits and rejections. No model or optimizer is changed
by this tool; each trainer adopts its own request between updates. A partial publication is journaled and
must be reconciled before a later revision. Input manifest: {schema: natlang.diloco-resident-source/1,
recipe: absolute remote path, stage, tokenizer, roles: [four ids], files: [{source: local absolute path,
target: remote absolute path, sha256, bytes}], geometry: optional remote evidence path}. The files must
include the complete immutable recipe, source, CE and geometry dependencies. Existing trainer seed bundles
must contain recipe-plan.json and the qualified geometry at the declared run resident paths.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
import cwctl
from natlang_neuralese.common import paths

SCHEMA = 'natlang.diloco-resident-source/1'
REPO = str(paths.root('repo'))
HELPER_SOURCE = Path(__file__).with_name('diloco_resident_worker.py')
CONTROL_DIRECTORY = '/work/.coreweave/resident-control'


def validate(record, manifest):
    if manifest.get('schema') != SCHEMA:
        raise ValueError('unknown resident source manifest')
    run = record['run']
    resident = run.get('resident')
    workers = record['learners']
    if not isinstance(resident, dict) or len(workers) != run['settings']['workers'] or any(not x for x in workers):
        raise ValueError('run lacks resident wiring or an assigned sandbox for every worker')
    if len(set(workers)) != len(workers):
        raise ValueError('each learner needs a distinct sandbox')
    if len(manifest.get('roles', [])) != 4 or not all(type(x) is int for x in manifest['roles']):
        raise ValueError('resident role IDs must be four integers')
    if not all(isinstance(manifest.get(k), str) and manifest[k] for k in ('recipe', 'stage', 'tokenizer')):
        raise ValueError('resident manifest needs recipe, stage and tokenizer')
    files = manifest.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('resident update needs explicit immutable files')
    targets = set()
    for item in files:
        if (type(item.get('bytes')) is not int or item['bytes'] < 0 or
                not isinstance(item.get('sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
            raise ValueError('file inventory needs a nonnegative byte count and SHA-256')
        source = Path(item['source'])
        target = Path(item['target'])
        if not source.is_absolute() or not target.is_absolute() or str(target) in targets:
            raise ValueError('file sources/targets must be absolute and targets unique')
        targets.add(str(target))
        if source.stat().st_size != item['bytes']:
            raise ValueError('local input size changed: ' + str(source))
        with source.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != item['sha256']:
            raise ValueError('local input hash changed: ' + str(source))
    if manifest['recipe'] not in targets:
        raise ValueError('immutable successor recipe must be in the file inventory')
    return resident, workers


def ensure_helper(sandbox):
    """Install an exact immutable control helper without touching the learner's runtime package."""
    content = HELPER_SOURCE.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    directory = CONTROL_DIRECTORY
    target, temporary = f'{directory}/{digest}.py', f'{directory}/{digest}.{os.getpid()}.part'
    result = sandbox.exec(['mkdir', '-p', directory]).result()
    if result.returncode:
        raise RuntimeError('cannot create resident helper directory')
    sandbox.write_file(temporary, content).result()
    code = ("import hashlib,os,sys; src,dst,want=sys.argv[1:]; "
            "actual=hashlib.sha256(open(src,'rb').read()).hexdigest(); "
            "assert actual==want, 'resident helper transfer mismatch'; "
            "exec(\"try: os.link(src,dst)\\nexcept FileExistsError: pass\"); "
            "assert hashlib.sha256(open(dst,'rb').read()).hexdigest()==want, 'resident helper target mismatch'; "
            "os.unlink(src)")
    result = sandbox.exec(['python3', '-c', code, temporary, target, digest]).result()
    if result.returncode:
        raise RuntimeError(f'resident helper verification failed: {result.stderr}')
    return target


def call(sandbox, args, helper=None):
    helper = helper or ensure_helper(sandbox)
    command = ['env', 'PYTHONPATH=' + REPO + '/training/neuralese:' + REPO,
               'python3', helper, *map(str, args)]
    result = sandbox.exec(command, cwd=REPO).result()
    if result.returncode:
        raise RuntimeError(f'resident host command failed ({result.returncode}): {result.stderr}')
    return json.loads(result.stdout)


def _write_journal(path, journal):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(journal, stream, sort_keys=True, indent=1)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def publish(record, manifest, journal_path, *, resume=False):
    resident, workers = validate(record, manifest)
    journal_path = Path(journal_path)
    helper_sha256 = hashlib.sha256(HELPER_SOURCE.read_bytes()).hexdigest()
    if journal_path.exists():
        if not resume:
            raise ValueError('publication journal exists; use --resume for its exact manifest and workers')
        journal = json.loads(journal_path.read_text())
        if (journal.get('schema') != 'natlang.diloco-resident-publication/1'
                or journal.get('run_id') != record['run']['id'] or journal.get('manifest') != manifest
                or journal.get('helper_sha256') != helper_sha256
                or [journal['workers'][str(i)]['sandbox_id'] for i in range(len(workers))] != workers):
            raise ValueError('resume journal does not bind this exact manifest and worker set')
        if journal['state'] == 'published':
            return journal
    else:
        if resume:
            raise ValueError('cannot resume without the original publication journal')
        journal = {'schema': 'natlang.diloco-resident-publication/1', 'run_id': record['run']['id'],
                   'manifest': manifest, 'helper_sha256': helper_sha256, 'state': 'staging', 'workers': {str(i): {'sandbox_id': s} for i, s in enumerate(workers)}}
    lock = threading.Lock()
    _write_journal(journal_path, journal)
    def update(i, **value):
        with lock:
            journal['workers'][str(i)].update(value)
            _write_journal(journal_path, journal)
    def prepare_worker(pair):
        i, sandbox_id = pair
        sandbox = cwctl.read_sandbox(sandbox_id)
        helper = ensure_helper(sandbox)
        existing = journal['workers'][str(i)].get('prepared')
        if existing is not None:
            return sandbox, existing, helper
        inbox = '/work/.coreweave/resident-staging'
        result = sandbox.exec(['mkdir', '-p', inbox]).result()
        if result.returncode: raise RuntimeError('cannot create resident staging directory')
        for item in manifest['files']:
            staged = f"{inbox}/{item['sha256']}-{i}.part"
            sandbox.write_file_streaming(staged, cwctl._chunks(Path(item['source'])), timeout_seconds=7200).result()
            call(sandbox, ['install', staged, item['target'], item['sha256'], item['bytes']], helper)
        args = ['prepare', resident['directory'], manifest['recipe'], manifest['stage'], manifest['tokenizer'], *manifest['roles']]
        if manifest.get('geometry'): args += ['--geometry', manifest['geometry']]
        prepared = call(sandbox, args, helper)
        update(i, state='prepared', prepared=prepared)
        return sandbox, prepared, helper
    try:
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            prepared = list(pool.map(prepare_worker, enumerate(workers)))
        identities = [item[1]['source_identity'] for item in prepared]
        if not identities[0].get('ce_sha256'):
            raise ValueError('resident source update must bind the CE supervision policy')
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError('workers prepared different logical source, recipe or CE bindings')
        journal['state'] = 'publishing'; journal['source_identity'] = identities[0]; _write_journal(journal_path, journal)
        for i, (sandbox, value, helper) in enumerate(prepared):
            if journal['workers'][str(i)].get('state') == 'published':
                continue
            result = call(sandbox, ['publish', resident['directory'], json.dumps(value['request'])], helper)
            update(i, state='published', publication=result)
        journal['state'] = 'published'; _write_journal(journal_path, journal)
    except Exception as error:
        journal['state'] = 'incomplete'; journal['error'] = f'{type(error).__name__}: {error}'
        _write_journal(journal_path, journal)
        raise
    return journal


def status(record):
    directory = record['run']['resident']['directory']
    result = {}
    for i, ident in enumerate(record['learners']):
        try:
            result[str(i)] = {'sandbox_id': ident, **call(cwctl.read_sandbox(ident), ['status', directory])}
        except Exception as error:
            result[str(i)] = {'sandbox_id': ident, 'availability': 'unavailable',
                              'error': f'{type(error).__name__}: {error}'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('publish-source-update'); p.add_argument('run_record', type=Path); p.add_argument('manifest', type=Path)
    p.add_argument('--journal', type=Path, required=True); p.add_argument('--resume', action='store_true')
    p = sub.add_parser('status'); p.add_argument('run_record', type=Path)
    args = parser.parse_args(); cwctl.load_env_file()
    record = json.loads(args.run_record.read_text())
    result = publish(record, json.loads(args.manifest.read_text()), args.journal, resume=args.resume) if args.action == 'publish-source-update' else status(record)
    print(json.dumps(result, sort_keys=True, indent=1))


if __name__ == '__main__':
    if not os.environ.get('NATLANG_CW_NO_REEXEC'): cwctl.reexec_in_venv()
    main()
