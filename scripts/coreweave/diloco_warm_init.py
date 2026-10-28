#!/usr/bin/env python3
"""Create a snapshot-backed DiLoCo owner with one validated dependency initialization.

The inline entrypoint initializes before starting the supervisor in the same PID 1.
The caller must supply the exact image, wheelhouse and runtime-file identities.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.pending')
    with temp.open('w') as stream:
        json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def prepare(config):
    work=Path(config['work']);repo=Path(config['repo']);runtime=Path(config['runtime'])
    for root in config['mirror_roots']:
        path=Path(root);target=work/'mirror'/root.lstrip('/')
        target.mkdir(parents=True,exist_ok=True);path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists() and not path.is_symlink():path.symlink_to(target,target_is_directory=True)
        if path.resolve()!=target.resolve():raise ValueError(f'mirror root differs: {path}')
    desc=json.loads((repo/'.coreweave-prebuilt/wheelhouse.json').read_text())
    archive=repo/'.coreweave-prebuilt/wheelhouse.tar.gz'
    if desc.get('kind')!='wheelhouse' or desc.get('sha256')!=config['wheelhouse_sha256'] or desc.get('image')!=config.get('wheelhouse_base_image',config['image']):
        raise ValueError('wheelhouse descriptor identity differs')
    if _sha(archive)!=config['wheelhouse_sha256']:raise ValueError('wheelhouse archive identity differs')
    if _sha(runtime/config['runtime_file'])!=config['runtime_sha256']:raise ValueError('runtime file identity differs')
    if not shutil.which('zstd'):
        subprocess.run(['apt-get','update','-qq'],check=True)
        subprocess.run(['apt-get','install','-y','-qq','--no-install-recommends','zstd'],check=True)
    if not shutil.which('zstd'):raise RuntimeError('zstd unavailable')
    image_ready=(os.environ.get('NATLANG_CW_IMAGE_READY')=='1' and Path('/opt/natlang/wheelhouse.sha256').is_file())
    if image_ready:
        if Path('/opt/natlang/wheelhouse.sha256').read_text().strip()!=config['wheelhouse_sha256']:
            raise ValueError('image wheelhouse attestation differs')
    else:
        subprocess.run([sys.executable,str(repo/'scripts/coreweave/prebuilt.py'),'install','--root',str(repo),'--work',str(work)],check=True)
    env={'NATLANG_ROW_INDEX_CACHE':str(work/'lfm3-row-index'),'HF_HUB_CACHE':str(work/'hf'),
         'PYTHONPATH':str(runtime/'training/neuralese')+':'+str(runtime),'PYTHONUNBUFFERED':'1'}
    for key in ('NATLANG_ROW_INDEX_CACHE','HF_HUB_CACHE'):Path(env[key]).mkdir(parents=True,exist_ok=True)
    # Fresh interpreter tests the environment that role children will inherit.
    modules=config.get('imports',['torch','transformers','safetensors','zstandard','natlang_neuralese.train.native_diloco'])
    subprocess.run([sys.executable,'-c','import importlib,json; [importlib.import_module(x) for x in '+repr(modules)+']; print(json.dumps({"imports":'+repr(modules)+'}))'],
                   env={**os.environ,**env},check=True)
    os.environ.update(env)
    receipt={'schema':'natlang.diloco-warm-initialization/1','passed':True,'image':config['image'],
             'wheelhouse_sha256':config['wheelhouse_sha256'],'runtime_sha256':config['runtime_sha256'],
             'runtime_file':str(runtime/config['runtime_file']),'mirror_roots':config['mirror_roots'],
             'environment':env,'zstd':shutil.which('zstd'),'imports':modules,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'pid':os.getpid(),'at':time.time()}
    _atomic(config['ready_path'],receipt)
    return receipt


def defaults(config):
    from natlang_neuralese.common.paths import root
    work=config.get('work','/work')
    repo=root('repo')
    return {'work':work,'repo':str(repo),'mirror_roots':[str(repo.parent),str(root('data_hdd'))],
            'role_control':work+'/recovery-role-control','ready_path':work+'/recovery-role-control/initialization.json',
            'input_wait_seconds':1800,
            **config}


def await_inputs(config):
    required=[Path(config['repo'])/'.coreweave-prebuilt/wheelhouse.json',
              Path(config['repo'])/'.coreweave-prebuilt/wheelhouse.tar.gz',
              Path(config['runtime'])/config['runtime_file']]
    # Snapshot mirror symlinks are created by prepare; use the underlying path
    # while waiting so a fresh container need not have its host roots yet.
    repo=Path(config['repo']);work=Path(config['work'])
    for root in config['mirror_roots']:
        try:relative=repo.relative_to(root)
        except ValueError:continue
        required[:2]=[work/'mirror'/root.lstrip('/')/relative/'.coreweave-prebuilt'/p.name for p in required[:2]]
        break
    deadline=time.monotonic()+config['input_wait_seconds']
    while not all(path.is_file() for path in required):
        if time.monotonic()>=deadline:raise TimeoutError('warm initialization inputs did not arrive')
        time.sleep(1)


def verify_readiness(receipt, owner, spec, *, boot_id, pid1_start_ticks):
    """Require initialization and the persistent PID 1 owner from the same boot."""
    config=defaults(spec['initialization'])
    work=Path(config['work']);runtime=Path(config['runtime'])
    expected_env={'NATLANG_ROW_INDEX_CACHE':str(work/'lfm3-row-index'),'HF_HUB_CACHE':str(work/'hf'),
                  'PYTHONPATH':str(runtime/'training/neuralese')+':'+str(runtime),'PYTHONUNBUFFERED':'1'}
    if (receipt.get('schema')!='natlang.diloco-warm-initialization/1' or receipt.get('passed') is not True
            or receipt.get('boot_id')!=boot_id or receipt.get('pid')!=1
            or receipt.get('image')!=spec['image'] or receipt.get('wheelhouse_sha256')!=config['wheelhouse_sha256']
            or receipt.get('runtime_sha256')!=config['runtime_sha256']
            or receipt.get('runtime_file')!=str(Path(config['runtime'])/config['runtime_file'])
            or receipt.get('mirror_roots')!=config['mirror_roots'] or receipt.get('environment')!=expected_env
            or not receipt.get('zstd') or receipt.get('imports')!=config.get('imports',[
                'torch','transformers','safetensors','zstandard','natlang_neuralese.train.native_diloco'])):
        raise ValueError('warm initialization receipt differs from declared runtime or boot')
    if (owner.get('schema')!='natlang.diloco-role-supervisor/1' or owner.get('boot_id')!=boot_id
            or owner.get('supervisor_pid')!=1 or not owner.get('supervisor_start_ticks')
            or owner.get('supervisor_start_ticks')!=pid1_start_ticks or not owner.get('owner_id')):
        raise ValueError('persistent role owner is not established on this boot')
    return {'ready':True,'boot_id':boot_id,'owner_id':owner['owner_id'],
            'supervisor_start_ticks':owner['supervisor_start_ticks'],'environment':receipt['environment']}


def owner_program(source, config):
    # Compilation happens on the operator before a sandbox can be allocated.
    source=source.replace('from __future__ import annotations\n','')
    compile(source,'<role-owner>','exec')
    module=Path(__file__).read_text()
    program=module+'\nCONFIG='+repr(config)+'\nSUPERVISOR='+repr(source)+'''\nif __name__ == '__main__':\n    Path(CONFIG['ready_path']).unlink(missing_ok=True)\n    await_inputs(CONFIG)\n    prepare(CONFIG)\n    sys.path.insert(0, str(Path(CONFIG['runtime']) / 'training/neuralese'))\n    sys.argv=['role-supervisor',CONFIG['role_control']]\n    exec(compile(SUPERVISOR,'<role-owner>','exec'),{'__name__':'__main__'})\n'''
    compile(program,'<warm-owner>','exec')
    return program


def validate_spec(spec):
    config=defaults(spec['initialization'])
    if (spec['image']!=config.get('image') or not re.search(r'@sha256:[0-9a-f]{64}$',spec['image'])
            or not spec.get('snapshot_id')):raise ValueError('snapshot and digest-pinned image are required')
    for key in ('wheelhouse_sha256','runtime_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}',config.get(key,'')):raise ValueError(f'invalid {key}')
    for key in ('work','repo','runtime','role_control','ready_path'):
        path=config.get(key,'')
        if not isinstance(path,str) or not path.startswith('/') or os.path.normpath(path)!=path:
            raise ValueError(f'{key} must be an absolute normalized path')
    relative=Path(config.get('runtime_file',''))
    if not relative.parts or relative.is_absolute() or '..' in relative.parts:raise ValueError('invalid runtime file')
    if (not config['mirror_roots'] or any(not isinstance(x,str) or not x.startswith('/') or os.path.normpath(x)!=x
                                          for x in config['mirror_roots'])):raise ValueError('invalid mirror roots')
    if config['input_wait_seconds']<=0:raise ValueError('input wait must be positive')
    if not any((service.get('endpoint') or {}).get('auth')=='share_token' for service in spec['services']):
        raise ValueError('warm service must use a persisted share token')
    return config


def create(spec, supervisor_source, intent_path, *, sandbox_run, cwctl):
    config=validate_spec(spec);program=owner_program(supervisor_source,config)
    intent=Path(intent_path)
    intent.parent.mkdir(parents=True,exist_ok=True)
    # Reserve before provider allocation; concurrent operators must never create
    # two sandboxes for the same durable intent.
    fd=os.open(intent,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    value={'schema':'natlang.diloco-warm-intent/1','phase':'intent','snapshot_id':spec['snapshot_id'],
           'image':spec['image'],'entrypoint_sha256':hashlib.sha256(program.encode()).hexdigest(),
           'entrypoint_compile_passed':True,'created_at':time.time()}
    _atomic(intent,value)
    sandbox=cwctl.throttled_create(lambda:sandbox_run('python','-u','-c',program,container_image=spec['image'],
        resources=spec['resources'],max_lifetime_seconds=spec['max_lifetime_seconds'],
        file_system_snapshot={'name':'work','mount_path':config['work'],'size':spec['volume_size'],
                              'file_system_snapshot_id':spec['snapshot_id']},
        tags=spec['tags'],services=spec['services'],data_plane_mode='gateway',auth=cwctl.sdk_auth()),attempts=1)
    value.update(sandbox_id=sandbox.sandbox_id,phase='created');_atomic(intent,value)
    token=Path(cwctl.keep_share_token(sandbox))
    with token.open('rb') as stream:os.fsync(stream.fileno())
    fd=os.open(token.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    value.update(phase='token-persisted',share_token_path=str(token));_atomic(intent,value)
    return sandbox,value


def wait_ready(sandbox, spec, intent_path, intent, *, timeout=1800):
    config=defaults(spec['initialization'])
    probe='''import hashlib,json,pathlib
p=pathlib.Path('/proc/1'); args=(p/'cmdline').read_bytes().split(b'\\0'); i=args.index(b'-c')
print(json.dumps({'entrypoint_sha256':hashlib.sha256(args[i+1]).hexdigest(),
 'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 'start_ticks':(p/'stat').read_text().rsplit(')',1)[1].split()[19],
 'receipt':json.loads(pathlib.Path(READY).read_text()),
 'owner':json.loads((pathlib.Path(CONTROL)/'state.json').read_text())}))'''
    code='READY='+repr(config['ready_path'])+'\nCONTROL='+repr(config['role_control'])+'\n'+probe
    compile(code,'<warm-readiness>','exec');deadline=time.monotonic()+timeout;last_error=''
    while time.monotonic()<deadline:
        try:
            result=sandbox.exec(['python3','-c',code],timeout_seconds=20).result(timeout=25)
            if result.returncode:raise ValueError('warm readiness probe failed')
            actual=json.loads(result.stdout)
            if actual['entrypoint_sha256']!=intent['entrypoint_sha256']:raise ValueError('PID 1 entrypoint differs')
            proof=verify_readiness(actual['receipt'],actual['owner'],spec,boot_id=actual['boot_id'],
                                   pid1_start_ticks=actual['start_ticks'])
            intent.update(phase='ready',readiness=proof);_atomic(intent_path,intent)
            return proof
        except (ValueError,KeyError,OSError,TimeoutError) as error:last_error=f'{type(error).__name__}: {error}'
        time.sleep(2)
    raise TimeoutError('warm initialization not ready: '+last_error)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--supervisor',type=Path,required=True);parser.add_argument('--intent',type=Path,required=True)
    args=parser.parse_args()
    from scripts.coreweave import cwctl
    from cwsandbox import Sandbox
    cwctl.load_env_file();cwctl.require_key()
    sandbox,value=create(json.loads(args.spec.read_text()),args.supervisor.read_text(),args.intent,sandbox_run=Sandbox.run,cwctl=cwctl)
    print(json.dumps({'sandbox_id':sandbox.sandbox_id,'phase':value['phase'],'entrypoint_sha256':value['entrypoint_sha256']}),flush=True)
    sandbox.wait(timeout=5400)
    proof=wait_ready(sandbox,json.loads(args.spec.read_text()),args.intent,value)
    print(json.dumps({'sandbox_id':sandbox.sandbox_id,'phase':'ready','readiness':proof}),flush=True)

# Executed only by the operator, not when embedded in the owner program.
if __name__=='__main__' and '__file__' in globals():main()
