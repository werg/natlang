#!/usr/bin/env python3
"""Verify a pinned recipe input closure on both development machines."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-service')
    parser.add_argument('--remote-storage')
    parser.add_argument('--expected-manifest-sha256', required=True)
    args = parser.parse_args()
    if args.wait_service:
        while True:
            result = subprocess.run(['systemctl', '--user', 'show', args.wait_service,
                                     '--property=ActiveState', '--property=Result',
                                     '--property=ExecMainStatus'], capture_output=True, text=True, check=True)
            state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
            if state.get('ActiveState') in {'active', 'activating', 'deactivating'}:
                time.sleep(15)
                continue
            if state.get('Result') != 'success' or state.get('ExecMainStatus') != '0':
                raise RuntimeError(f'priority transfer did not complete successfully: {state}')
            break
    if args.remote_storage:
        command = ['python3', args.remote_root + '/scripts/link_development_data.py',
                   '--repo', args.remote_root, '--storage', args.remote_storage]
        subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
                        shlex.join(command)], check=True)
    if digest(args.manifest) != args.expected_manifest_sha256:
        raise ValueError('priority manifest hash mismatch')
    manifest = json.loads(args.manifest.read_text())
    inputs = manifest['inputs']
    if len(inputs) != manifest['input_count'] or len({item['path'] for item in inputs}) != len(inputs):
        raise ValueError('manifest input count or duplicate path mismatch')
    local_errors = []
    for item in inputs:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('input outside checkout')
        if not (args.root / relative).resolve().is_relative_to(args.root.resolve()):
            raise ValueError(f'source input escapes checkout: {relative}')
        if digest(args.root / relative) != item['sha256']:
            local_errors.append(item['path'])
    if local_errors:
        raise ValueError(f'recipe source input hashes changed: {local_errors}')
    remote = """import json,sys,hashlib
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['root']);checked=[];errors=[]
allowed=[root.resolve()]+([Path(p['storage']).resolve()] if p.get('storage') else [])
for item in p['inputs']:
 relative=Path(item['path'])
 assert not relative.is_absolute() and '..' not in relative.parts
 target=root/relative
 assert any(target.resolve().is_relative_to(base) for base in allowed), 'input escapes approved roots'
 try:
  h=hashlib.sha256()
  with target.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
  actual=h.hexdigest()
  if actual!=item['sha256']: errors.append({'path':item['path'],'actual':actual,'expected':item['sha256']})
  else: checked.append({'path':item['path'],'sha256':actual,'bytes':target.stat().st_size})
 except OSError as error: errors.append({'path':item['path'],'error':str(error)})
print(json.dumps({'checked':checked,'errors':errors}))
"""
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
                             'python3 -c ' + shlex.quote(remote)],
                            input=json.dumps({'root': args.remote_root, 'storage': args.remote_storage, 'inputs': inputs}),
                            text=True, capture_output=True, check=True)
    checked = json.loads(result.stdout)
    report = {'schema': 'natlang.development_pipeline_input_verification/1',
              'status': 'passed' if not checked['errors'] and len(checked['checked']) == len(inputs) else 'failed',
              'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'manifest_path': str(args.manifest), 'manifest_sha256': digest(args.manifest),
              'host': args.host, 'remote_root': args.remote_root,
              'input_count': len(inputs), 'source_hashes_verified': len(inputs), **checked,
              'scope': 'Exact recipe input bytes; does not grant new data admission or mark the entire pipeline mirror complete.'}
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({'status': report['status'], 'input_count': len(inputs),
                      'verified': len(checked['checked']), 'errors': len(checked['errors'])}))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
