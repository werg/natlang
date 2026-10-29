#!/usr/bin/env python3
"""Low-impact private export of a replicated DiLoCo coordinator's sealed state.

The replicated coordinator is authoritative as an immutable generation baseline plus
a hash-chained journal; replaying the sealed journal prefix over the baseline
reproduces the exact coordinator (fragments with theta/outer momentum/EMA, pending
windows, quorum timers, sessions, sequence/head). Restore reassembles exactly the
`natlang.diloco-replica-seed/1` directory that `replica_seed.verify_seed` and
`install_seed` accept.

The coordinator hosts have one CPU and sit on the replication commit path, so the
sandbox does as little as possible (2026-10-29 v5 fence):

* No sandbox hashing. Integrity comes from the coordinator's own sealed digests
  (`replica-baseline.json` per file, `record.json` payload digests); the workstation
  verifies every byte against them during its owner GET.
* No staging write. Each large immutable file is PUT once, straight from its inode, as
  a private object named by its sealed SHA-256; a blob already in the cloud is never
  sent again. Small files (< 1 MiB: receipts, records, progress, manifest) travel in
  the selection read and are hashed on the workstation.
* SCHED_IDLE, idle I/O class, nice 19 and a token-bucket rate cap (8 MB/s default).
  The uploader opens every selected file before the first PUT, so generation
  retirement cannot remove bytes mid-export; no hardlinks or leases are taken.
* Gated: a cycle starts only after a new outer merge, with no unregistered rotated
  generation, journal payload below the rotation threshold and no submit/tick within
  the quiet window; before every file the uploader re-checks and pauses (bounded,
  then aborts; uploaded blobs remain reusable).
* A cycle is verified only when every blob is GET-verified on the workstation into
  the local blob store and the small-file manifest is privately exported with owner
  GET. Cycles never overlap and start at least `interval_seconds` apart.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

SEED_SCHEMA = 'natlang.diloco-replica-seed/1'
EXPORT_SCHEMA = 'natlang.diloco-replica-export/2'
JOURNAL_SCHEMA = 'natlang.diloco-replica-export-journal/2'
SMALL = 1 << 20
BLOCK = 1 << 20
RESTARTED = 'interrupted before verification (supervisor restart)'


def identity(value):
    """Exactly replica_journal._identity: the chain hash of a record or identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# ---------------------------------------------------------------- sandbox side (stdlib only)

class RotationInProgress(RuntimeError):
    pass


def _active(store):
    store = Path(store); path = store / 'primary-active.json'
    if not path.exists(): path = store / 'replica-active.json'
    pointer = json.loads(path.read_bytes())
    retention = store / 'retention.json'
    registered = {x['id'] for x in json.loads(retention.read_text())['generations']} if retention.exists() else set()
    present = {p.name[len('generation-'):] for p in store.glob('generation-*') if p.is_dir()}
    rotating = (pointer.get('id') != 'initial' and pointer.get('id') not in registered) or bool(present - registered)
    return pointer, rotating


def _tail_activity(journal, count=24):
    """Wall time of the newest submit/tick among the last records (merges run in these)."""
    names = sorted(p.name for p in Path(journal).glob('[0-9]' * 20))[-count:]
    newest = None
    for name in names:
        try: event = json.loads((Path(journal) / name / 'record.json').read_bytes())['event']
        except (OSError, ValueError, KeyError): continue
        if event.get('kind') in ('submit', 'tick', 'close', 'completion') and isinstance(event.get('wall'), (int, float)):
            newest = max(newest or 0, event['wall'])
    return newest


def gate(store, *, quiet_seconds, rotation_bytes):
    """Cheap state check: small files and directory listings only."""
    pointer, rotating = _active(store)
    journal = Path(pointer['journal']); ident = json.loads((journal / 'identity.json').read_bytes())
    payload = 0; count = 0
    for directory in journal.glob('[0-9]' * 20):
        try: payload += json.loads((directory / 'record.json').read_bytes())['payload_bytes']; count += 1
        except (OSError, ValueError, KeyError): pass
    activity = _tail_activity(journal); now = time.time()
    reasons = []
    if rotating: reasons.append('rotation in progress')
    if payload > rotation_bytes * ident['max_payload_bytes'] or count + 64 > ident['max_transactions']: reasons.append('rotation imminent')
    if activity is not None and now - activity < quiet_seconds: reasons.append('merge activity')
    return {'open': not reasons, 'reasons': reasons, 'generation': pointer.get('id'), 'payload_bytes': payload, 'records': count,
            'last_activity_wall': activity, 'now': now}


def select(plan):
    """Sealed selection without hashing; small files are returned, large ones listed with their sealed digest."""
    store = Path(plan['store']); pointer, rotating = _active(store)
    if rotating: raise RotationInProgress('coordinator generation rotation in progress')
    baseline = Path(pointer['baseline']); journal = Path(pointer['journal'])
    receipt_raw = (baseline / 'replica-baseline.json').read_bytes(); value = json.loads(receipt_raw)
    identity_raw = (journal / 'identity.json').read_bytes(); ident = json.loads(identity_raw)
    if ident['baseline_sha256'] != pointer['baseline_sha256'] or identity(ident) != pointer['head']:
        raise ValueError('journal identity does not bind the active pointer')
    small = {'baseline/replica-baseline.json': receipt_raw, 'journal/identity.json': identity_raw}; large = []
    def add(relative, path, size, digest):
        st = os.stat(path)
        if st.st_size != size: raise ValueError('sealed size differs: ' + relative)
        if size < SMALL: small[relative] = Path(path).read_bytes()
        else: large.append({'path': relative, 'source': str(path), 'bytes': size, 'sha256': digest,
                            'stat': [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]})
    for item in value['files']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('invalid baseline path')
        add('baseline/' + item['path'], baseline / relative, item['bytes'], item['sha256'])
    previous = identity(ident); records = []
    while True:
        directory = journal / f'{len(records) + 1:020d}'
        try: raw = (directory / 'record.json').read_bytes()
        except FileNotFoundError: break
        record = json.loads(raw)
        if record.get('sequence') != len(records) + 1 or record.get('previous') != previous: raise ValueError('journal chain mismatch')
        previous = identity(record); records.append((raw, record, previous))
    records = records[:len(records) - len(records) % 2]
    for raw, record, _ in records:
        name = f"journal/{record['sequence']:020d}"; small[name + '/record.json'] = raw
        add(name + '/payload', journal / f"{record['sequence']:020d}" / 'payload', record['payload_bytes'], record['payload_sha256'])
    progress = json.loads((Path(pointer['work_root']) / 'progress.json').read_bytes()) if pointer.get('work_root') else {}
    rounds = progress.get('rounds') or []
    out = {'pointer': pointer, 'sequence': len(records), 'head': records[-1][2] if records else identity(ident),
           'small': {k: base64.b64encode(v).decode() for k, v in small.items()}, 'large': large,
           'last_round_at': rounds[-1].get('at') if rounds else None, 'selected_at': time.time()}
    Path(plan['output']).write_text(json.dumps(out))
    return {'output': plan['output'], 'small': len(small), 'large': len(large), 'sequence': out['sequence']}


def _rate_limited(handle, size, rate, on_block):
    sent = 0; start = time.monotonic()
    while sent < size:
        block = handle.read(min(BLOCK, size - sent))
        if not block: raise ValueError('source ended before its sealed size')
        sent += len(block)
        if hasattr(os, 'posix_fadvise'): os.posix_fadvise(handle.fileno(), 0, sent, os.POSIX_FADV_DONTNEED)
        on_block()
        ahead = sent / rate - (time.monotonic() - start)
        if ahead > 0: time.sleep(ahead)
        yield block


def upload(plan):
    """Open every selected inode first, then PUT each at the capped rate between gate checks."""
    import requests
    capability = json.loads(Path(plan['capability']).read_text())
    handles = {}
    try:
        for item in capability['items']:
            handle = open(item['source'], 'rb'); st = os.fstat(handle.fileno())
            if [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns] != item['stat']: raise ValueError('selected inode changed: ' + item['path'])
            handles[item['sha256']] = handle
        for item in capability['items']:
            waited = 0.0
            while True:
                state = gate(plan['store'], quiet_seconds=plan['quiet_seconds'], rotation_bytes=plan['rotation_bytes'])
                if state['open']: break
                if waited >= plan['max_pause_seconds']:
                    print(json.dumps({'event': 'aborted', 'reasons': state['reasons'], 'sha256': item['sha256']}), flush=True); return 3
                time.sleep(15); waited += 15
            handle = handles[item['sha256']]; handle.seek(0); start = time.time()
            body = _rate_limited(handle, item['bytes'], plan['rate_bytes'], lambda: None)
            response = requests.put(item['url'], headers={**item['headers'], 'Content-Length': str(item['bytes'])}, data=body, timeout=600)
            st = os.fstat(handle.fileno())
            if [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns] != item['stat']: raise ValueError('inode changed during upload: ' + item['path'])
            print(json.dumps({'event': 'uploaded', 'sha256': item['sha256'], 'status': response.status_code,
                              'seconds': time.time() - start}), flush=True)
            if response.status_code not in (200, 201, 204): return 4
        return 0
    finally:
        for handle in handles.values(): handle.close()


# ---------------------------------------------------------------- workstation side

def build_seed(selection):
    """Check the chain from the returned small files and return the seed receipt bytes."""
    from natlang_neuralese.common.hashing import sha256_hex
    small = {k: base64.b64decode(v) for k, v in selection['small'].items()}
    files = {}
    baseline_raw = small['baseline/replica-baseline.json']
    if sha256_hex(baseline_raw) != selection['pointer']['baseline_sha256']: raise ValueError('baseline receipt digest differs')
    baseline = json.loads(baseline_raw); large = {x['path']: x for x in selection['large']}
    files['baseline/replica-baseline.json'] = (len(baseline_raw), sha256_hex(baseline_raw))
    for item in baseline['files']:
        relative = 'baseline/' + item['path']
        if relative in small:
            if (len(small[relative]), sha256_hex(small[relative])) != (item['bytes'], item['sha256']): raise ValueError('small baseline file differs: ' + relative)
        elif (large[relative]['bytes'], large[relative]['sha256']) != (item['bytes'], item['sha256']): raise ValueError('large binding differs: ' + relative)
        files[relative] = (item['bytes'], item['sha256'])
    ident_raw = small['journal/identity.json']; ident = json.loads(ident_raw); previous = identity(ident)
    if ident['baseline_sha256'] != selection['pointer']['baseline_sha256']: raise ValueError('journal identity differs')
    files['journal/identity.json'] = (len(ident_raw), sha256_hex(ident_raw))
    for sequence in range(1, selection['sequence'] + 1):
        name = f'journal/{sequence:020d}'; raw = small[name + '/record.json']; record = json.loads(raw)
        if record['sequence'] != sequence or record['previous'] != previous: raise ValueError('journal chain mismatch')
        payload = name + '/payload'
        if payload in small:
            if (len(small[payload]), sha256_hex(small[payload])) != (record['payload_bytes'], record['payload_sha256']): raise ValueError('payload differs')
        elif (large[payload]['bytes'], large[payload]['sha256']) != (record['payload_bytes'], record['payload_sha256']): raise ValueError('payload binding differs')
        files[name + '/record.json'] = (len(raw), sha256_hex(raw)); files[payload] = (record['payload_bytes'], record['payload_sha256'])
        previous = identity(record)
    if previous != selection['head'] or selection['sequence'] % 2: raise ValueError('sealed head differs')
    receipt = {'schema': SEED_SCHEMA, 'run_id': baseline['run_id'], 'baseline_sha256': selection['pointer']['baseline_sha256'],
               'sequence': selection['sequence'], 'head': selection['head'], 'lease': None, 'freeze_nonce': 'read-only-export',
               'files': [{'path': k, 'bytes': b, 'sha256': d} for k, (b, d) in sorted(files.items())]}
    return json.dumps(receipt, sort_keys=True, indent=1).encode(), small


def write_manifest(path, receipt_raw, small, export):
    with tarfile.open(path, 'x') as tar:
        def add(name, raw):
            info = tarfile.TarInfo(name); info.size = len(raw); info.mtime = 0; info.mode = 0o600; tar.addfile(info, io.BytesIO(raw))
        add('seed/seed-receipt.json', receipt_raw)
        for name, raw in sorted(small.items()): add('seed/' + name, raw)
        add('export.json', json.dumps(export, sort_keys=True, indent=1).encode())


def restore(journal_path, export_id, destination):
    """Reassemble the seed directory from the manifest and the local blob store; check every file."""
    state = json.loads(Path(journal_path).read_text())
    entry = next(e for e in state['exports'] if e['export_id'] == export_id and e.get('phase') == 'verified')
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(entry['manifest_local'], 'r') as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.startswith('seed/') or '..' in Path(member.name).parts: continue
            target = destination / member.name[5:]; target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open('xb') as dst: shutil.copyfileobj(src, dst)
    receipt = json.loads((destination / 'seed-receipt.json').read_text())
    for item in receipt['files']:
        target = destination / item['path']
        if not target.exists():
            blob = state['blobs'][item['sha256']]; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(blob['local'], target)
        with target.open('rb') as f:
            if target.stat().st_size != item['bytes'] or hashlib.file_digest(f, 'sha256').hexdigest() != item['sha256']:
                raise ValueError('restored file differs: ' + item['path'])
    return {'destination': str(destination), 'sequence': receipt['sequence'], 'head': receipt['head'], 'files': len(receipt['files']),
            'seed_receipt_sha256': hashlib.sha256((destination / 'seed-receipt.json').read_bytes()).hexdigest()}


def prune_blobs(state, journal, *, deletion_log, keep_export_id):
    """Remove local blobs the newest verified export does not reference; cloud objects stay."""
    keep = set(next(e for e in state['exports'] if e['export_id'] == keep_export_id)['blob_shas'])
    removed = []
    for digest, blob in list(state['blobs'].items()):
        if digest in keep or not blob.get('local') or not Path(blob['local']).exists(): continue
        with open(deletion_log, 'a') as log:
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} delete {blob['local']} sha256={digest} bytes={blob['bytes']} "
                      f"reason=local coordinator blob unreferenced by newest verified export {keep_export_id}; private object {blob['object']['name']} retained\n")
        Path(blob['local']).unlink(); blob['local'] = None; blob['local_pruned_at'] = time.time(); removed.append(digest)
        _write(journal, state)
    return removed


CW_PYTHON = Path.home() / '.local/share/natlang/cw-venv/bin/python'
CODE = ('scripts/__init__.py', 'scripts/coreweave/diloco_replica_export.py')
LOW = ['nice', '-n', '19', 'chrt', '-i', '0', 'ionice', '-c', '3']
RUNNER = ('import json,sys;sys.path[:0]=[sys.argv[1],sys.argv[2]];import scripts.coreweave.diloco_replica_export as m;'
          'r=getattr(m,sys.argv[3])(json.loads(open(sys.argv[4]).read()));print(json.dumps(r));sys.exit(r if isinstance(r,int) else 0)')
compile(RUNNER, 'runner', 'exec')


def remote_argv(code, runtime, function, plan, python='python3'):
    if function not in ('gate', 'select', 'upload'): raise ValueError('unknown remote step')
    return [*LOW, python, '-c', RUNNER, code, runtime, function, plan]


def _dgxops():
    from natlang_neuralese.common.paths import resolve
    return resolve('data_nvme', 'dgxops')


def _write(path, value):
    from natlang_neuralese.common.jsonio import write_json_durable
    write_json_durable(path, value)


def _gql(query, variables):
    import requests
    r = requests.post('https://api.wandb.ai/graphql', auth=('api', os.environ['WANDB_API_KEY']), json={'query': query, 'variables': variables}, timeout=60)
    r.raise_for_status(); data = r.json()
    if data.get('errors'): raise RuntimeError(str(data['errors'])[:500])
    return data['data']


def _blob_run(config):
    registry = Path(config['state_root']) / config['blob_run_registry']
    if registry.exists(): return json.loads(registry.read_text())['run_id']
    os.environ['WANDB_DIR'] = str(_dgxops() / 'coreweave-wandb'); import wandb
    run = wandb.init(entity='werg', project='natlang', job_type='coreweave-production-export', name=config['role_prefix'] + '-blobs',
                     settings=wandb.Settings(console='off', disable_code=True, disable_git=True, x_disable_stats=True, quiet=True))
    run_id = run.id; run.finish(); _write(registry, {'run_id': run_id, 'project': 'werg/natlang', 'access': 'PRIVATE'})
    return run_id


def _exec(box, argv, timeout):
    result = box.exec(argv, timeout_seconds=timeout).result(timeout=timeout + 60)
    if result.returncode not in (0, 3): raise RuntimeError(f'remote step failed ({result.returncode}): {result.stderr[-1500:]}')
    return result


def cycle(config, state, journal):
    import cwctl, requests, wandb
    from natlang_neuralese.common.hashing import sha256_hex
    from private_export_guard import commit_uploaded_run_file, owner_readback_reserve_bytes
    from verified_download import verify_download_blocks
    cwctl.load_env_file(); box = cwctl.read_sandbox(config['sandbox_id'], patience=300)
    staging = config['staging'].rstrip('/'); runtime = config['remote_runtime'].rstrip('/') + '/training/neuralese'
    repo = Path(__file__).resolve().parents[2]
    for relative in CODE:
        raw = (repo / relative).read_bytes(); compile(raw, relative, 'exec'); box.write_file(f'{staging}/code/{relative}', raw).result(timeout=120)
    def remote(function, plan, timeout):
        path = f'{staging}/{function}-plan.json'; box.write_file(path, json.dumps(plan).encode()).result(timeout=60)
        return _exec(box, remote_argv(f'{staging}/code', runtime, function, path), timeout)
    gate_plan = {'store': config['store'], 'quiet_seconds': config['quiet_seconds'], 'rotation_bytes': config['rotation_fraction']}
    state_now = json.loads(remote('gate', gate_plan, 300).stdout.strip().splitlines()[-1])
    last_round = max((e.get('last_round_at') or 0 for e in state['exports'] if e.get('phase') == 'verified'), default=0)
    if not state_now['open']: return {'skipped': state_now['reasons']}
    export_id = time.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    entry = {'export_id': export_id, 'phase': 'selecting', 'started_at': time.time(), 'gate': state_now}
    state['exports'].append(entry); _write(journal, state)
    remote('select', {'store': config['store'], 'output': f'{staging}/selection.json'}, 900)
    selection = json.loads(box.read_file(f'{staging}/selection.json').result(timeout=300))
    if (selection.get('last_round_at') or 0) <= last_round:
        entry.update(phase='skipped', reason='no outer merge since the last verified export'); _write(journal, state); return entry
    receipt_raw, small = build_seed(selection); blobs = state.setdefault('blobs', {})
    needed = {x['sha256']: x for x in selection['large']}
    missing = [x for d, x in needed.items() if d not in blobs]  # uploaded-but-unverified blobs are only re-read
    intent = {'schema': 'natlang.diloco-replica-export-intent/2', 'export_id': export_id, 'sandbox_id': config['sandbox_id'],
              'generation': selection['pointer'].get('id'), 'baseline_sha256': selection['pointer']['baseline_sha256'],
              'sequence': selection['sequence'], 'head': selection['head'], 'seed_receipt_sha256': sha256_hex(receipt_raw),
              'blobs': {d: x['bytes'] for d, x in needed.items()}, 'upload': {d: needed[d]['bytes'] for d in (x['sha256'] for x in missing)},
              'last_round_at': selection['last_round_at'], 'at': time.time()}
    intent_path = Path(config['state_root']) / f'coordinator-v5-export-{export_id}-intent.json'; _write(intent_path, intent)
    entry.update(phase='uploading', intent=str(intent_path), sequence=selection['sequence'], head=selection['head'],
                 generation=intent['generation'], blob_shas=sorted(needed), last_round_at=selection['last_round_at'],
                 upload_bytes=sum(intent['upload'].values())); _write(journal, state)
    run_id = _blob_run(config); run_path = 'werg/natlang/' + run_id
    assert _gql('query { project(name:"natlang",entityName:"werg") { access } }', {})['project']['access'] == 'PRIVATE'
    if missing:
        mutation = ('mutation CreateRunFiles($entity:String!,$project:String!,$run:String!,$files:[String!]!) {createRunFiles(input:'
                    '{entityName:$entity,projectName:$project,runName:$run,files:$files}) {runID uploadHeaders files {name uploadUrl}}}')
        items = []
        for start in range(0, len(missing), 50):
            batch = missing[start:start + 50]
            data = _gql(mutation, {'entity': 'werg', 'project': 'natlang', 'run': run_id, 'files': ['blobs/' + x['sha256'] for x in batch]})['createRunFiles']
            urls = {f['name']: f['uploadUrl'] for f in data['files']}; headers = dict(h.split(':', 1) for h in data['uploadHeaders'])
            items += [{**x, 'url': urls['blobs/' + x['sha256']], 'headers': headers} for x in batch]
        cap = f'{staging}/upload-capability-{export_id}.json'
        box.write_file(cap, json.dumps({'items': items}).encode()).result(timeout=120); box.exec(['chmod', '600', cap], timeout_seconds=60).result(timeout=90)
        try:
            result = remote('upload', {'capability': cap, 'store': config['store'], 'quiet_seconds': config['quiet_seconds'],
                                       'rotation_bytes': config['rotation_fraction'], 'rate_bytes': config['rate_bytes'],
                                       'max_pause_seconds': config['max_pause_seconds']}, 6 * 3600)
        finally:
            box.exec(['rm', '-f', '--', cap], timeout_seconds=60).result(timeout=90)
        events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"event"')]
        entry['upload_events'] = events[-5:]; _write(journal, state)
        uploaded = {e['sha256'] for e in events if e['event'] == 'uploaded' and e['status'] in (200, 201, 204)}
        api = wandb.Api()
        for digest in uploaded:
            commit_uploaded_run_file(api, run_path, 'blobs/' + digest)
            blobs[digest] = {'bytes': needed[digest]['bytes'], 'object': {'run': run_path, 'name': 'blobs/' + digest}, 'verified': False, 'local': None}
        _write(journal, state)
        if result.returncode != 0: raise RuntimeError('upload stopped: ' + json.dumps(events[-1:] or result.stderr[-500:]))
    store = Path(config['local_root']) / 'blobs'; store.mkdir(parents=True, exist_ok=True); api = wandb.Api()
    entry['phase'] = 'owner-get'; _write(journal, state)
    for digest in sorted(needed):
        blob = blobs[digest]
        if blob.get('verified') and blob.get('local') and Path(blob['local']).exists(): continue
        remote_file = api.run(blob['object']['run']).file(blob['object']['name'])
        if remote_file.size != blob['bytes']:
            del blobs[digest]; _write(journal, state)  # the next cycle uploads it again
            raise RuntimeError('private blob missing or wrong size: ' + digest)
        url = remote_file.url
        response = requests.get(url, auth=('api', os.environ['WANDB_API_KEY']), stream=True, timeout=60, allow_redirects=True)
        with response:
            response.raise_for_status()
            target = store / digest; target.unlink(missing_ok=True)
            check = verify_download_blocks(response.iter_content(8 << 20), {'sha256': digest, 'bytes': blob['bytes']}, destination=target,
                                           reserve_bytes=int(config.get('owner_reserve_bytes', owner_readback_reserve_bytes())))
        copy = check.get('local_copy') or {}
        if not copy.get('committed'): raise RuntimeError('local blob copy not committed: ' + json.dumps(copy))
        blob.update(verified=True, local=copy['path'], verified_at=time.time()); _write(journal, state)
    manifest = Path(config['local_root']) / 'manifests' / f'{export_id}.tar'; manifest.parent.mkdir(parents=True, exist_ok=True)
    export = {'schema': EXPORT_SCHEMA, 'export_id': export_id, 'seed_receipt_sha256': intent['seed_receipt_sha256'],
              'pointer': selection['pointer'], 'sequence': selection['sequence'], 'head': selection['head'],
              'blobs': {d: blobs[d]['object'] for d in sorted(needed)}, 'last_round_at': selection['last_round_at']}
    staged = manifest.with_suffix('.staged.tar'); staged.unlink(missing_ok=True); write_manifest(staged, receipt_raw, small, export)
    role = f"{config['role_prefix']}-manifest-{export_id[-8:]}"
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    env = {**os.environ, 'CW_EXPORT_LOCAL': '1', 'CW_EXPORT_VERIFY_ON_OWNER': '1', 'CW_EXPORT_OWNER_DESTINATION': str(manifest),
           'CW_EXPORT_EXPECTED_SHA256': digest, 'CW_EXPORT_EXPECTED_BYTES': str(staged.stat().st_size),
           'CW_EXPORT_OWNER_RESERVE_BYTES': str(config.get('owner_reserve_bytes', 40 << 30))}
    log = Path(config['state_root']) / f'coordinator-v5-export-{export_id}.log'
    with log.open('ab') as output:
        code = subprocess.run([str(CW_PYTHON), str(_dgxops() / 'coreweave-private-runfile-export-20261014.py'), role, str(staged)],
                              env=env, stdout=output, stderr=subprocess.STDOUT).returncode
    receipt_path = _dgxops() / f'coreweave-production-{role}-private-export-receipt-20261014.json'
    if code != 0 or not receipt_path.exists(): raise RuntimeError(f'manifest export failed ({code}); see {log}')
    receipt = json.loads(receipt_path.read_text()); check = receipt['independent_verification']
    if check.get('readback_location') != 'owner' or check.get('sha256') != digest or not (check.get('local_copy') or {}).get('committed'):
        raise RuntimeError('manifest lacks owner GET and local copy')
    staged.unlink()
    entry.update(phase='verified', manifest_local=str(manifest), manifest_object={'run': receipt['run'], 'name': receipt['name']},
                 manifest_sha256=digest, manifest_receipt=str(receipt_path), verified_at=time.time()); _write(journal, state)
    if config.get('prune_superseded_local', True):
        entry['pruned'] = prune_blobs(state, journal, deletion_log=config['deletion_log'], keep_export_id=export_id); _write(journal, state)
    box.exec(['rm', '-f', '--', f'{staging}/selection.json'], timeout_seconds=60).result(timeout=90)
    return entry


def run(config_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    config = json.loads(Path(config_path).read_text())
    journal = Path(config['state_root']) / config['journal_name']
    state = json.loads(journal.read_text()) if journal.exists() else {'schema': JOURNAL_SCHEMA, 'exports': [], 'blobs': {}}
    if state.get('disabled') or not config.get('enabled'):
        raise SystemExit('export not enabled: ' + json.dumps(state.get('disabled') or 'config.enabled is false'))
    for entry in state['exports']:
        if entry.get('phase') not in ('verified', 'failed', 'skipped'): entry.update(phase='failed', error=RESTARTED)
    _write(journal, state)
    while True:
        counted = [e for e in state['exports'] if e.get('error') != RESTARTED and e.get('phase') != 'skipped']
        last = max((e['started_at'] for e in counted), default=0)
        failed = bool(counted) and counted[-1].get('phase') != 'verified'
        wait = (config['retry_seconds'] if failed else config['interval_seconds']) - (time.time() - last)
        if wait > 0: time.sleep(min(wait, 300)); continue
        try:
            result = cycle(config, state, journal)
            print(json.dumps({k: result.get(k) for k in ('export_id', 'phase', 'sequence', 'skipped', 'reason')}), flush=True)
            if result.get('skipped') or result.get('phase') == 'skipped': time.sleep(config['gate_poll_seconds'])
        except Exception as error:
            entry = state['exports'][-1] if state['exports'] else None
            if entry is not None and entry.get('phase') not in ('verified', 'skipped'):
                entry.update(phase='failed', error=f'{type(error).__name__}: {str(error)[:1500]}'); _write(journal, state)
            print('export cycle failed', type(error).__name__, str(error)[:500], flush=True); time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run'); r.add_argument('config')
    s = sub.add_parser('restore'); s.add_argument('journal'); s.add_argument('export_id'); s.add_argument('destination')
    a = parser.parse_args()
    if a.action == 'run': return run(a.config)
    print(json.dumps(restore(a.journal, a.export_id, a.destination), indent=1))


if __name__ == '__main__':
    main()
