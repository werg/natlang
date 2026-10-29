"""Owned, checkpoint-preserving restoration of an independently exported cohort.

This controller never allocates sandboxes. Every input is bound by the verified
staging manifest; continuation preserves the prior outer session and each rank's
own full checkpoint. A new outer requires the separate reanchor protocol.
"""
from __future__ import annotations
import hashlib,json,os,uuid
from pathlib import Path
from natlang_neuralese.common.jsonio import write_json_durable


def _sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _chunks(path):
    with Path(path).open('rb') as f:
        while block:=f.read(4*1024*1024):yield block


def stage_worker_checkpoints(staging, hosts, *, cwctl):
    """Transfer each distinct, verified full worker state to its own host slot.

    Final slots are immutable hardlinks created only after the sandbox has
    independently read and verified all bytes. Interrupted writes are never
    accepted; retries reuse an exact verified final slot.
    """
    staging=Path(staging).resolve();manifest=json.loads((staging/'staging.json').read_text())
    if manifest.get('schema')!='natlang.diloco-staged-recovery/1' or not manifest.get('replica_baseline_staged'):
        raise ValueError('replicated verified cohort staging is required')
    if sorted(hosts)!=list(range(4)):raise ValueError('four distinct worker roles are required')
    components=manifest['components'];results={};seen=set()
    verification=json.loads((staging/'replica-restored'/'verified-restore.json').read_text())
    closure=staging/'diloco-replica-recovery'/'recovery-closure.json'
    if (verification.get('schema')!='natlang.diloco-verified-restore/1' or _sha(closure)!=verification.get('receipt_sha256') or
        verification.get('worker_sha256')!={str(i):components['worker-'+str(i)]['sha256'] for i in range(4)}):
        raise ValueError('staged full-state verification differs')
    for rank in range(4):
        host=hosts[rank];sandbox=host['sandbox_id'];slot=host['checkpoint_slot']
        if sandbox in seen or not slot.startswith('/work/') or '..' in Path(slot).parts:raise ValueError('worker destination is ambiguous')
        seen.add(sandbox);local=staging/manifest['worker_staged'][str(rank)]/'checkpoint.pt';identity=components['worker-'+str(rank)]
        if local.stat().st_size!=identity['bytes'] or _sha(local)!=identity['sha256']:raise ValueError('staged worker checkpoint changed')
        box=cwctl.read_sandbox(sandbox,patience=60)
        check=('import hashlib,json,pathlib,sys\np=pathlib.Path(sys.argv[1]);sha=sys.argv[2];size=int(sys.argv[3])\n'
               'if not p.exists():print("missing");sys.exit(0)\n'
               'if p.is_symlink() or not p.is_file() or p.stat().st_size!=size or hashlib.file_digest(p.open("rb"),"sha256").hexdigest()!=sha:raise ValueError("destination checkpoint differs")\n'
               'print("verified")\n')
        result=box.exec(['python3','-c',check,slot,identity['sha256'],str(identity['bytes'])],timeout_seconds=7200).result(timeout=7230)
        if result.returncode:raise ValueError('existing worker checkpoint failed verification')
        if result.stdout.strip()!='verified':
            temporary=slot+'.upload-'+uuid.uuid4().hex
            mkdir=box.exec(['mkdir','-p',str(Path(slot).parent)],timeout_seconds=30).result(timeout=60)
            if mkdir.returncode:raise ValueError('worker staging directory unavailable')
            box.write_file_streaming(temporary,_chunks(local),timeout_seconds=7200).result(timeout=7230)
            program=('import hashlib,os,pathlib,sys\np=pathlib.Path(sys.argv[1]);q=pathlib.Path(sys.argv[2]);sha=sys.argv[3];size=int(sys.argv[4])\n'
                     'if p.is_symlink() or not p.is_file() or p.stat().st_size!=size or hashlib.file_digest(p.open("rb"),"sha256").hexdigest()!=sha:raise ValueError("worker upload differs")\n'
                     'with p.open("rb") as f:os.fsync(f.fileno())\nos.link(p,q);p.unlink();fd=os.open(q.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)\n')
            result=box.exec(['python3','-c',program,temporary,slot,identity['sha256'],str(identity['bytes'])],timeout_seconds=7200).result(timeout=7230)
            if result.returncode:raise ValueError('worker upload was not verified and published')
        results[str(rank)]={'sandbox_id':sandbox,'path':slot,'sha256':identity['sha256'],'bytes':identity['bytes'],'global_step':identity['global_step']}
    return results


def stage_replica_seed(seed, hosts, *, cwctl, archive_path):
    """Install the same verified, sealed recovery seed on two owned hosts."""
    import tarfile
    from natlang_neuralese.train.diloco.replica_seed import verify_seed
    seed=Path(seed).resolve();receipt_sha=_sha(seed/'seed-receipt.json');record=verify_seed(seed,expected_receipt_sha256=receipt_sha)
    if len(hosts)!=2 or len({x['sandbox_id'] for x in hosts})!=2:raise ValueError('two distinct replica hosts are required')
    archive=Path(archive_path)
    if archive.exists():
        expected={'seed/'+item['path']:item['sha256'] for item in record['files']}
        expected['seed/seed-receipt.json']=receipt_sha;seen={}
        with tarfile.open(archive) as old:
            for member in old:
                if member.isdir():continue
                if not member.isfile() or member.name not in expected or member.name in seen:raise ValueError('existing seed archive inventory differs')
                seen[member.name]=hashlib.file_digest(old.extractfile(member),'sha256').hexdigest()
        if seen!=expected:raise ValueError('existing seed archive differs')
    else:
        archive.parent.mkdir(parents=True,exist_ok=True)
        temp=archive.with_name(archive.name+'.partial-'+uuid.uuid4().hex)
        try:
            with tarfile.open(temp,'w',dereference=True) as out:out.add(seed,arcname='seed',filter=lambda member:None if member.name.endswith('/.writer.lock') else member)
            os.link(temp,archive)
        finally:temp.unlink(missing_ok=True)
    archive_sha=_sha(archive);size=archive.stat().st_size;result=[]
    for host in hosts:
        root=host['restore_root']
        if not root.startswith('/work/') or '..' in Path(root).parts:raise ValueError('invalid replica restore root')
        box=cwctl.read_sandbox(host['sandbox_id'],patience=60)
        remote=root+'/seed-'+archive_sha+'.tar';partial=remote+'.partial-'+uuid.uuid4().hex
        check=('import hashlib,pathlib,sys\np=pathlib.Path(sys.argv[1]);s=sys.argv[2];n=int(sys.argv[3])\n'
               'if not p.exists():print("missing");sys.exit(0)\n'
               'if p.is_symlink() or p.stat().st_size!=n or hashlib.file_digest(p.open("rb"),"sha256").hexdigest()!=s:raise ValueError("seed archive differs")\nprint("verified")\n')
        probe=box.exec(['python3','-c',check,remote,archive_sha,str(size)],timeout_seconds=7200).result(timeout=7230)
        if probe.returncode:raise ValueError('existing replica seed archive differs')
        if probe.stdout.strip()!='verified':
            made=box.exec(['mkdir','-p',root],timeout_seconds=30).result(timeout=60)
            if made.returncode:raise ValueError('replica staging directory unavailable')
            box.write_file_streaming(partial,_chunks(archive),timeout_seconds=7200).result(timeout=7230)
            publish=('import hashlib,os,pathlib,sys\np=pathlib.Path(sys.argv[1]);q=pathlib.Path(sys.argv[2]);s=sys.argv[3];n=int(sys.argv[4])\n'
                     'if p.is_symlink() or p.stat().st_size!=n or hashlib.file_digest(p.open("rb"),"sha256").hexdigest()!=s:raise ValueError("seed upload differs")\n'
                     'with p.open("rb") as f:os.fsync(f.fileno())\nos.link(p,q);p.unlink();fd=os.open(q.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)\n')
            uploaded=box.exec(['python3','-c',publish,partial,remote,archive_sha,str(size)],timeout_seconds=7200).result(timeout=7230)
            if uploaded.returncode:raise ValueError('replica seed upload failed verification')
        # Each extraction is private and no peer starts until the seed verifies.
        script=('import hashlib,json,os,pathlib,shutil,sys,tarfile,uuid\n'
                'from natlang_neuralese.train.diloco.replica_seed import verify_seed,install_seed\n'
                'from natlang_neuralese.common.jsonio import write_json_durable\n'
                'a=pathlib.Path(sys.argv[1]);d=pathlib.Path(sys.argv[2]);store=pathlib.Path(sys.argv[3]);sha=sys.argv[4]\n'
                'if not d.exists():\n'
                ' tmp=d.with_name(d.name+".partial-"+uuid.uuid4().hex);tmp.mkdir(parents=True)\n'
                ' with tarfile.open(a) as t:\n'
                '  for m in t:\n'
                '   p=pathlib.PurePosixPath(m.name)\n'
                '   if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0]!="seed" or not (m.isfile() or m.isdir()):raise ValueError("unsafe seed member")\n'
                '   t.extract(m,tmp,filter="data")\n'
                ' verify_seed(tmp/"seed",expected_receipt_sha256=sha);os.rename(tmp,d)\n'
                'verify_seed(d/"seed",expected_receipt_sha256=sha)\n'
                'intent=store.parent/(store.name+".install-intent.json");binding={"seed_receipt_sha256":sha,"store":str(store)}\n'
                'if intent.exists() and json.loads(intent.read_text())!=binding:raise ValueError("seed install intent differs")\n'
                'if store.exists() and not (store/"seed-install.json").exists():\n'
                ' if not intent.exists() or store.is_symlink():raise ValueError("unowned incomplete seed store")\n'
                ' shutil.rmtree(store)\n'
                'if not intent.exists():write_json_durable(intent,binding)\n'
                'if not store.exists():install_seed(d/"seed",store,expected_receipt_sha256=sha)\n'
                'freeze={"schema":"natlang.diloco-replica-file-freeze/1","nonce":json.loads((d/"seed"/"seed-receipt.json").read_text())["freeze_nonce"]};fp=store/"replica-files-frozen.json"\n'
                'if fp.exists() and json.loads(fp.read_text())!=freeze:raise ValueError("file custody freeze differs")\n'
                'if not fp.exists():write_json_durable(fp,freeze)\n'
                'v=json.loads((store/"seed-install.json").read_text())\n'
                'if v.get("seed_receipt_sha256")!=sha:raise ValueError("installed seed differs")\n'
                'print(json.dumps(v))\n')
        extraction=root+'/extract-'+archive_sha;store=host['store']
        if not store.startswith('/work/') or '..' in Path(store).parts or not Path(store).is_relative_to(root) or store==root:raise ValueError('invalid replica store')
        cmd=['env','PYTHONPATH='+host['runtime']+'/training/neuralese',host['python'],'-c',script,remote,extraction,store,receipt_sha]
        installed=box.exec(cmd,timeout_seconds=7200).result(timeout=7230)
        if installed.returncode:raise ValueError('replica seed installation failed')
        value=json.loads(installed.stdout)
        if value.get('head')!=record['head'] or value.get('sequence')!=record['sequence'] or value.get('run_id')!=record['run_id']:
            raise ValueError('replica seed installation identity differs')
        result.append({'sandbox_id':host['sandbox_id'],'store':store,'seed_receipt_sha256':receipt_sha,'head':record['head'],'sequence':record['sequence']})
    return {'archive_sha256':archive_sha,'archive_bytes':size,'replicas':result}


def start_restored_workers(directory, hosts, pins, *, nonce, coordinator_role, cwctl, fences):
    """Start exact rank-specific saved states under the released nested owners."""
    from scripts.coreweave.diloco_replica_controller import capture_worker_ownership
    if sorted(hosts)!=list(range(4)) or sorted(pins)!=list(map(str,range(4))):raise ValueError('four exact workers required')
    record=json.loads((Path(directory)/'promotion.json').read_text())
    if record.get('phase') not in ('workers-released','released') or record['intent']['activation_nonce']!=nonce:
        raise ValueError('primary readiness and owned release are required')
    for rank in range(4):
        host=hosts[rank];pin=pins[str(rank)];argv=host['argv'];digest=hashlib.sha256(json.dumps(argv,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        if (host['sandbox_id']!=pin['sandbox_id'] or host['worker_argv_sha256']!=digest or argv.count('--out')!=1 or
            Path(argv[argv.index('--out')+1])/'checkpoint.pt'!=Path(pin['path'])):
            raise ValueError('worker argv does not bind its exact full checkpoint slot')
        if argv.count('--diloco')!=1 or '--diloco-new-outer' in argv:raise ValueError('restore must continue the existing outer session')
        settings=json.loads(argv[argv.index('--diloco')+1]);worker_id=settings.get('worker_id')
        if worker_id is None:worker_id=int(host.get('env',{}).get('NATLANG_DILOCO_WORKER_ID','-1'))
        if (worker_id!=rank or settings.get('workers')!=4 or settings.get('run_id')!=record['intent']['run_id'] or
                settings.get('coordinator_url')!=record['intent']['new_url']):
            raise ValueError('worker rank or restored coordinator session differs')
        env=host.get('env',{});secret_dir=env.get('NATLANG_CW_SECRETS','')
        if not secret_dir.startswith('/work/') or env.get('NATLANG_DILOCO_TOKEN') or env.get('NATLANG_DILOCO_GATEWAY_TOKEN'):
            raise ValueError('restored worker needs scoped credential files')
        if any('COREWEAVE' in key.upper() or key.upper().startswith('CW_') for key in host.get('env',{})):
            raise ValueError('provider credentials cannot enter worker environment')
        try:
            observed=fences.sdk_capture_live_worker(host,nonce)
            if observed['worker_argv_sha256']!=digest:raise ValueError('existing worker argv differs')
            continue
        except ValueError:
            pass
        box=cwctl.read_sandbox(host['sandbox_id'],patience=60)
        check=('import hashlib,pathlib,sys\np=pathlib.Path(sys.argv[1]);assert p.is_file() and not p.is_symlink() and p.stat().st_size==int(sys.argv[3]) and hashlib.file_digest(p.open("rb"),"sha256").hexdigest()==sys.argv[2]\n')
        checked=box.exec(['python3','-c',check,pin['path'],pin['sha256'],str(pin['bytes'])],timeout_seconds=7200).result(timeout=7230)
        if checked.returncode:raise ValueError('worker pin changed before start')
        state_result=box.exec(['cat',host['directory'].rstrip('/')+'/state.json'],timeout_seconds=30).result(timeout=60)
        if state_result.returncode:raise ValueError('worker owner state unavailable')
        state=json.loads(state_result.stdout)
        if state.get('owner_id')!=host['owner_id'] or state.get('boot_id')!=host['boot_id'] or state.get('pending_start') or state.get('roles',{}).get('worker',{}).get('alive'):
            raise ValueError('worker owner changed or has an unverified live worker')
        request={'schema':'natlang.diloco-role-request/1','id':'restore-'+nonce[:24]+'-'+str(rank),
                 'owner_id':host['owner_id'],'expected_generation':state['generation'],'action':'start','role':'worker',
                 'argv':argv,'env':host.get('env',{})}
        publisher=host['publisher'];program=('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
            'p=Path(sys.argv[1]);assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]\n'
            'print(json.dumps(runpy.run_path(str(p))["submit"](sys.argv[3],json.loads(sys.argv[4]),timeout=60)))\n')
        result=box.exec(['python3','-c',program,publisher['path'],publisher['sha256'],host['directory'],json.dumps(request)],timeout_seconds=90).result(timeout=120)
        if result.returncode or json.loads(result.stdout).get('accepted') is not True:raise ValueError('worker owner refused exact restored role')
        fences.sdk_capture_live_worker(host,nonce)
    return capture_worker_ownership(directory,coordinator_role=coordinator_role,hosts=hosts,nonce=nonce)


def start_restored_peer(host, *, nonce, seed_record, cwctl, fences, token, gateway=None):
    """Start an authenticated peer as a fenced owner child and verify its head."""
    import time
    from natlang_neuralese.train.diloco.replica_transport import witness_head
    box=cwctl.read_sandbox(host['sandbox_id'],patience=60);root=host['restore_root'];store=host['store']
    config={'baseline':store+'/baseline','baseline_sha256':seed_record['baseline_sha256'],'store':store,
            'work_root':host['work_root'],'host':host['listen_host'],'port':host['port'],'token_file':host['token_file'],
            'max_transaction_bytes':host['max_transaction_bytes']}
    raw=json.dumps(config,sort_keys=True,separators=(',',':')).encode();digest=hashlib.sha256(raw).hexdigest();path=root+'/peer-'+digest+'.json'
    box.write_file(path,raw).result(timeout=60)
    argv=[host['python'],'-m','natlang_neuralese.train.diloco.replica_cli','peer','--config',path,'--config-sha256',digest]
    state_result=box.exec(['cat',host['directory'].rstrip('/')+'/state.json'],timeout_seconds=30).result(timeout=60)
    if state_result.returncode:raise ValueError('peer owner state unavailable')
    state=json.loads(state_result.stdout)
    if state.get('owner_id')!=host['owner_id'] or state.get('boot_id')!=host['boot_id'] or state.get('pending_start'):
        raise ValueError('peer owner identity changed')
    role_name=host.get('peer_role','actor');live=state.get('roles',{}).get(role_name,{})
    argv_sha=hashlib.sha256(json.dumps(argv,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    if live.get('alive'):
        events=[e for e in state.get('events',[]) if e.get('event')=='start' and e.get('role')==role_name and e.get('pid')==live.get('pid')]
        if len(events)!=1 or events[0].get('argv_sha256')!=argv_sha:raise ValueError('different peer already owns the role')
        started={'pid':live['pid'],'process_start_ticks':live['start_ticks'],'owner_id':host['owner_id'],'generation':state['generation']}
    else:
        request={'schema':'natlang.diloco-role-request/1','id':'restore-peer-'+nonce[:24]+'-'+host['sandbox_id'],
            'owner_id':host['owner_id'],'expected_generation':state['generation'],'action':'start','role':role_name,'promotion_nonce':nonce,
            'argv':argv,'env':{'PYTHONPATH':host['runtime']+'/training/neuralese'}}
        role={'sandbox_id':host['sandbox_id'],'directory':host['directory'],'boot_id':host['boot_id'],'runtime':host['runtime'],'request':request}
        fences.sdk_authorize_seed_server(role,nonce)
        publisher=host['publisher'];program=('import hashlib,json,runpy,sys\nfrom pathlib import Path\n'
            'p=Path(sys.argv[1]);assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]\n'
            'print(json.dumps(runpy.run_path(str(p))["submit"](sys.argv[3],json.loads(sys.argv[4]),timeout=60)))\n')
        response=box.exec(['python3','-c',program,publisher['path'],publisher['sha256'],host['directory'],json.dumps(request)],timeout_seconds=90).result(timeout=120)
        if response.returncode:raise ValueError('peer owner start failed')
        started=json.loads(response.stdout)
        if started.get('accepted') is not True:raise ValueError('peer owner rejected start')
    preserved={'role':role_name,'pid':started['pid'],'start_ticks':started['process_start_ticks'],'argv_sha256':argv_sha}
    fences.sdk_record_preserved_role(host,nonce,preserved)
    expected={'run_id':seed_record['run_id'],'baseline_sha256':seed_record['baseline_sha256'],'sequence':seed_record['sequence'],'head':seed_record['head']}
    deadline=time.monotonic()+host.get('readiness_timeout',300)
    while True:
        try:
            actual=witness_head(host['url'],token,allow_http=host.get('allow_http',False),gateway=gateway)
            if actual!=expected:raise ValueError('restored peer witnessed a different state')
            break
        except OSError:
            if time.monotonic()>=deadline:raise
            time.sleep(1)
    stop={'schema':'natlang.diloco-role-request/1','id':'restore-peer-stop-'+nonce[:24]+'-'+host['sandbox_id'],
          'owner_id':host['owner_id'],'expected_generation':started['generation'],'role':role_name,'action':'stop',
          'expected_pid':started['pid'],'expected_process_start_ticks':started['process_start_ticks']}
    return {'sandbox_id':host['sandbox_id'],'directory':host['directory'],'publisher':host['publisher'],'request':stop,
            'preserve_role':preserved,'witness':expected}




def verify_runtime_prerequisites(hosts, *, cwctl):
    """Qualify preinstalled, owned runtime environments before any transfer."""
    result={}
    for rank,host in sorted(hosts.items()):
        spec=host.get('runtime_prerequisite')
        if (not isinstance(spec,dict) or set(spec)!={'python_sha256','packages'} or
                not isinstance(spec['packages'],dict) or not {'torch','transformers','safetensors','accelerate'}<=set(spec['packages']) or
                not all(isinstance(k,str) and isinstance(v,str) and v for k,v in spec['packages'].items())):
            raise ValueError('exact preinstalled Python and package prerequisite required')
        program=('import hashlib,importlib.metadata,json,pathlib,sys\n'
                 's=json.loads(sys.argv[1]);p=pathlib.Path(sys.executable).resolve()\n'
                 'if hashlib.file_digest(p.open("rb"),"sha256").hexdigest()!=s["python_sha256"]:raise ValueError("python executable differs")\n'
                 'actual={k:importlib.metadata.version(k) for k in s["packages"]}\n'
                 'if actual!=s["packages"]:raise ValueError("runtime package versions differ")\n'
                 'import torch,transformers,safetensors,accelerate\nprint(json.dumps({"python":str(p),"packages":actual}))\n')
        box=cwctl.read_sandbox(host['sandbox_id'],patience=60)
        checked=box.exec([host['python'],'-c',program,json.dumps(spec,sort_keys=True)],timeout_seconds=120).result(timeout=150)
        if checked.returncode:raise ValueError('owned host runtime prerequisite failed')
        result[str(rank)]={'sandbox_id':host['sandbox_id'],'python_sha256':spec['python_sha256'],'packages':spec['packages']}
    return result


def stage_scoped_credential(host, *, local_file, expected_sha256, remote_file, cwctl):
    """Install a scoped credential through SDK file transfer, never argv or logs."""
    import stat
    local=Path(local_file);before=local.stat();secret=local.read_bytes();after=local.stat()
    if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=0o600 or
            (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or
            hashlib.sha256(secret).hexdigest()!=expected_sha256 or not secret or secret!=secret.strip() or not remote_file.startswith('/work/') or os.path.normpath(remote_file)!=remote_file):
        raise ValueError('scoped credential source or destination differs')
    box=cwctl.read_sandbox(host['sandbox_id'],patience=60)
    check=('import hashlib,pathlib,stat,sys\np=pathlib.Path(sys.argv[1])\n'
           'if not p.exists():print("missing");sys.exit(0)\n'
           'if p.is_symlink() or not p.is_file() or stat.S_IMODE(p.stat().st_mode)!=0o600 or hashlib.sha256(p.read_bytes()).hexdigest()!=sys.argv[2]:raise ValueError("scoped credential differs")\nprint("verified")\n')
    observed=box.exec(['python3','-c',check,remote_file,expected_sha256],timeout_seconds=30).result(timeout=60)
    if observed.returncode:raise ValueError('existing scoped credential differs')
    if observed.stdout.strip()!='verified':
        temporary=remote_file+'.pending-'+uuid.uuid4().hex+'/value'
        create=('import os,pathlib,sys\np=pathlib.Path(sys.argv[1]);p.parent.parent.mkdir(parents=True,exist_ok=True);os.mkdir(p.parent,0o700);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)\n')
        made=box.exec(['python3','-c',create,temporary],timeout_seconds=30).result(timeout=60)
        if made.returncode:raise ValueError('credential private file creation failed')
        box.write_file(temporary,secret).result(timeout=60)
        publish=('import hashlib,os,pathlib,stat,sys\np=pathlib.Path(sys.argv[1]);q=pathlib.Path(sys.argv[2])\n'
                 'if p.is_symlink() or stat.S_IMODE(p.stat().st_mode)!=0o600 or hashlib.sha256(p.read_bytes()).hexdigest()!=sys.argv[3]:raise ValueError("scoped credential transfer differs")\n'
                 'with p.open("rb") as f:os.fsync(f.fileno())\nos.link(p,q);p.unlink();p.parent.rmdir();fd=os.open(q.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)\n')
        sent=box.exec(['python3','-c',publish,temporary,remote_file,expected_sha256],timeout_seconds=30).result(timeout=60)
        if sent.returncode:raise ValueError('scoped credential transfer was not published')
    return {'sandbox_id':host['sandbox_id'],'path':remote_file,'sha256':expected_sha256}


def stage_worker_dependencies(staging, hosts, *, cwctl):
    """Install every authenticated logical dependency and verify the runtime."""
    staging=Path(staging).resolve();manifest=json.loads((staging/'staging.json').read_text());index_path=staging/'dependencies'/'dependency-index.json'
    if _sha(index_path)!=manifest.get('dependency_index_sha256'):raise ValueError('dependency index changed')
    index=json.loads(index_path.read_text())
    if index.get('schema')!='natlang.diloco-recovery-dependencies/1':raise ValueError('invalid dependency index')
    items={}
    for item in index['bindings']:
        logical=item['path'];sha=item['sha256'];size=item['bytes']
        if logical.startswith('@'):continue
        if not logical.startswith('/') or os.path.normpath(logical)!=logical:raise ValueError('invalid logical dependency path')
        prior=items.get(logical)
        if prior and (prior['sha256']!=sha or prior['bytes']!=size):raise ValueError('logical dependency collision')
        if 'external' in item:
            source=(manifest.get('external_dependencies') or {}).get(logical,{}).get('resolved_local_path')
            if not source:raise ValueError('external recovery dependency is unavailable')
        else:
            if item.get('object')!='objects/'+sha:raise ValueError('invalid dependency object binding')
            source=staging/'dependencies'/item['object']
        items[logical]={'source':Path(source),'sha256':sha,'bytes':size}
    parent=manifest['parent'];parent_file=staging/parent['staged']
    if parent_file.stat().st_size!=parent['bytes'] or _sha(parent_file)!=parent['sha256']:raise ValueError('staged parent differs')
    verified=set()
    for item in items.values():
        key=(str(item['source']),item['sha256'],item['bytes'])
        if key not in verified:
            if item['source'].stat().st_size!=item['bytes'] or _sha(item['source'])!=item['sha256']:raise ValueError('dependency object differs')
            verified.add(key)
    result={}
    for rank,host in sorted(hosts.items()):
        argv=host['argv']
        if argv.count('--continue-from')!=1:raise ValueError('worker lacks original lineage parent')
        parent_path=argv[argv.index('--continue-from')+1]
        if not parent_path.startswith('/') or os.path.normpath(parent_path)!=parent_path:raise ValueError('invalid original parent path')
        own=dict(items);existing=own.get(parent_path)
        if existing and (existing['sha256']!=parent['sha256'] or existing['bytes']!=parent['bytes']):raise ValueError('parent dependency collides')
        own[parent_path]={'source':parent_file,'sha256':parent['sha256'],'bytes':parent['bytes']}
        box=cwctl.read_sandbox(host['sandbox_id'],patience=60)
        for logical,item in sorted(own.items()):
            check=('import hashlib,pathlib,sys\np=pathlib.Path(sys.argv[1]);s=sys.argv[2];n=int(sys.argv[3])\n'
                   'if not p.exists():print("missing");sys.exit(0)\n'
                   'if p.is_symlink() or not p.is_file() or p.stat().st_size!=n or hashlib.file_digest(p.open("rb"),"sha256").hexdigest()!=s:raise ValueError("dependency differs")\nprint("verified")\n')
            check_result=box.exec(['python3','-c',check,logical,item['sha256'],str(item['bytes'])],timeout_seconds=7200).result(timeout=7230)
            if check_result.returncode:raise ValueError('existing dependency differs: '+logical)
            if check_result.stdout.strip()=='verified':continue
            temporary=logical+'.recovery-'+uuid.uuid4().hex
            made=box.exec(['mkdir','-p',str(Path(logical).parent)],timeout_seconds=30).result(timeout=60)
            if made.returncode:raise ValueError('dependency directory unavailable')
            box.write_file_streaming(temporary,_chunks(item['source']),timeout_seconds=7200).result(timeout=7230)
            publish=('import hashlib,os,pathlib,sys\np=pathlib.Path(sys.argv[1]);q=pathlib.Path(sys.argv[2]);s=sys.argv[3];n=int(sys.argv[4])\n'
                     'if p.is_symlink() or p.stat().st_size!=n or hashlib.file_digest(p.open("rb"),"sha256").hexdigest()!=s:raise ValueError("dependency upload differs")\n'
                     'with p.open("rb") as f:os.fsync(f.fileno())\nos.link(p,q);p.unlink();fd=os.open(q.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)\n')
            done=box.exec(['python3','-c',publish,temporary,logical,item['sha256'],str(item['bytes'])],timeout_seconds=7200).result(timeout=7230)
            if done.returncode:raise ValueError('dependency upload failed: '+logical)
        expected=manifest['components']['worker-'+str(rank)]['runtime_code_sha256']
        program=('import sys\nfrom pathlib import Path\nfrom natlang_neuralese.train.native_resident import code_inventory,digest\n'
                 'assert digest(code_inventory(Path(sys.argv[1])))==sys.argv[2]\n')
        code=box.exec(['env','PYTHONPATH='+host['runtime']+'/training/neuralese',host['python'],'-c',program,
            host['runtime']+'/training/neuralese/natlang_neuralese',expected],timeout_seconds=180).result(timeout=210)
        if code.returncode:raise ValueError('worker runtime inventory differs')
        result[str(rank)]={'sandbox_id':host['sandbox_id'],'dependencies':len(own),'runtime_code_sha256':expected}
    return result


def run_full_restore(directory, config):
    """Restore one previously exported outer; never reanchor or allocate hosts."""
    import fcntl
    from scripts.coreweave import cwctl,diloco_replica_fence as fences
    from scripts.coreweave.diloco_replica_controller import run_coreweave_promotion
    from natlang_neuralese.train.diloco.replica_recovery import prepare_restored_peer_seed
    from natlang_neuralese.train.diloco.replica_seed import verify_seed
    if config.get('mode')!='continue-existing-outer':raise ValueError('new outer requires the separate reanchor protocol')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'restore.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=json.dumps(config,sort_keys=True,separators=(',',':')).encode();binding=hashlib.sha256(raw).hexdigest()
        journal=directory/'restore.json'
        if journal.exists():
            state=json.loads(journal.read_text())
            if state.get('config_sha256')!=binding:raise ValueError('restore journal belongs to another configuration')
        else:
            state={'schema':'natlang.diloco-full-restore-controller/1','config_sha256':binding,'phase':'intent'};write_json_durable(journal,state)
        promotion=config['promotion'];intent=promotion['intent'];nonce=intent['activation_nonce'];staging=Path(config['staging']).resolve()
        manifest_path=staging/'staging.json'
        if _sha(manifest_path)!=config['staging_manifest_sha256']:raise ValueError('staging manifest differs from controller binding')
        manifest=json.loads(manifest_path.read_text());verified=json.loads((staging/'replica-restored'/'verified-restore.json').read_text())
        if (manifest['coordinator_run_id']!=intent['run_id'] or verified['owner_journal_sha256']!=promotion['supervisor_proof']['journal']['sha256']):
            raise ValueError('staged cohort and supervisor ownership differ')
        workers={int(k):v for k,v in config['worker_hosts'].items()};peers=config['peer_hosts']
        if sorted(workers)!=list(range(4)) or len(peers)!=2:raise ValueError('four workers and two replica owners required')
        if {x['sandbox_id'] for x in peers}-{x['sandbox_id'] for x in workers.values()}:raise ValueError('replica owners must be authenticated worker hosts')
        for peer in peers:
            match=[v for v in workers.values() if v['sandbox_id']==peer['sandbox_id']]
            if len(match)!=1 or any(peer.get(k)!=match[0].get(k) for k in ('directory','owner_id','boot_id','runtime')):
                raise ValueError('replica owner binding differs from authenticated worker owner')
        cwctl.load_env_file();cwctl.require_key()
        replacements={int(k):v for k,v in promotion['full_restore']['replacements'].items()}
        if state['phase']=='intent':
            evidence=fences.verify_full_restore_fence(journal_ref=promotion['supervisor_proof']['journal'],expected_primary=promotion['expected_primary'],
                expected_workers=promotion['expected_workers'],replacements=replacements,nonce=nonce,get_sandbox=fences.sdk_get_sandbox,observe_owner=fences.sdk_observe_replacement_owner)
            authority=[{k:v[k] for k in ('sandbox_id','directory','owner_id','boot_id','runtime')} for v in workers.values()]
            fences.sdk_set_worker_authority(authority,nonce=nonce,action='fence')
            state.update(phase='fenced',fence=evidence);write_json_durable(journal,state)
        if state['phase']=='fenced':
            prerequisites=verify_runtime_prerequisites(workers,cwctl=cwctl)
            dependencies=stage_worker_dependencies(staging,workers,cwctl=cwctl)
            pins=stage_worker_checkpoints(staging,workers,cwctl=cwctl);state.update(phase='workers-staged',pins=pins,dependencies=dependencies,runtime_prerequisites=prerequisites);write_json_durable(journal,state)
        if state['phase']=='workers-staged':
            seed=directory/'seed';seed_store=directory/'seed-store';closure=staging/'diloco-replica-recovery'
            if not (seed/'seed-receipt.json').is_file():
                import shutil
                for incomplete in (seed,seed_store):
                    if incomplete.is_symlink():raise ValueError('restore scratch is a symlink')
                    if incomplete.exists():shutil.rmtree(incomplete)
                prepared=prepare_restored_peer_seed(closure=closure,restored=staging/'replica-restored',destination=seed_store,seed=seed,
                    receipt_sha256=verified['receipt_sha256'],freeze_nonce=nonce)
                seed_sha=prepared['receipt_sha256']
            else:seed_sha=_sha(seed/'seed-receipt.json')
            seed_record=verify_seed(seed,expected_receipt_sha256=seed_sha)
            if (seed_record['run_id']!=intent['run_id'] or seed_record['sequence']!=intent['sequence'] or seed_record['head']!=intent['head']):
                raise ValueError('restored peer seed differs from exact promotion intent')
            transferred=stage_replica_seed(seed,peers,cwctl=cwctl,archive_path=directory/'seed.tar')
            state.update(phase='seeds-staged',seed=seed_record,seed_sha256=seed_sha,transfer=transferred);write_json_durable(journal,state)
        if state['phase']=='seeds-staged':
            old_sha=config['old_token_sha256'];new_sha=intent['new_token_sha256']
            if _sha(config['old_token_file'])!=old_sha or _sha(promotion['outgoing_replica']['old_token_file'])!=old_sha or _sha(promotion['new_token_file'])!=new_sha:
                raise ValueError('scoped credential bindings differ')
            for host in peers:stage_scoped_credential(host,local_file=config['old_token_file'],expected_sha256=old_sha,remote_file=host['token_file'],cwctl=cwctl)
            stage_scoped_credential(peers[0],local_file=promotion['new_token_file'],expected_sha256=new_sha,remote_file=promotion['activation_config']['token_file'],cwctl=cwctl)
            if promotion['activation_config']['token_file']==peers[0]['token_file']:raise ValueError('old and new coordinator token paths must differ')
            for rank,host in sorted(workers.items()):
                env=host.get('env',{});secret_dir=env.get('NATLANG_CW_SECRETS','')
                if (not secret_dir.startswith('/work/') or os.path.normpath(secret_dir)!=secret_dir or
                        env.get('NATLANG_DILOCO_TOKEN') or env.get('NATLANG_DILOCO_GATEWAY_TOKEN')):
                    raise ValueError('workers require private secret files, not token environment variables')
                stage_scoped_credential(host,local_file=promotion['new_token_file'],expected_sha256=new_sha,remote_file=secret_dir+'/diloco-token',cwctl=cwctl)
                gateway=host.get('gateway_credential')
                if not isinstance(gateway,dict) or set(gateway)!={'local_file','sha256'}:raise ValueError('exact worker gateway credential required')
                stage_scoped_credential(host,local_file=gateway['local_file'],expected_sha256=gateway['sha256'],remote_file=secret_dir+'/diloco-gateway',cwctl=cwctl)
            token=Path(config['old_token_file']).read_text().strip();started=[]
            for host in peers:started.append(start_restored_peer(host,nonce=nonce,seed_record=state['seed'],cwctl=cwctl,fences=fences,token=token,gateway=host.get('gateway')))
            state.update(phase='peers-ready',peers=started);write_json_durable(journal,state)
        if state['phase']=='peers-ready':
            source=state['peers'][0];primary=promotion['primary_role']
            activation=promotion['activation_config'];seed=state['seed'];outgoing=promotion['outgoing_replica']
            if (activation.get('baseline')!=peers[0]['store']+'/baseline' or activation.get('journal')!=peers[0]['store']+'/journal' or
                    activation.get('file_custody_store')!=peers[0]['store'] or activation.get('baseline_sha256')!=seed['baseline_sha256'] or
                    outgoing.get('baseline_sha256')!=seed['baseline_sha256'] or outgoing.get('url')!=peers[1]['url']):
                raise ValueError('promotion does not bind both exact restored replicas')
            if source['sandbox_id']!=primary['sandbox_id'] or source['directory']!=primary['directory']:
                raise ValueError('promoted replica must be the coordinator owner')
            promo=json.loads(json.dumps(promotion));promo['promoted_replica_role']={k:source[k] for k in ('sandbox_id','directory','publisher','request')}
            preserve={p['sandbox_id']:p['preserve_role'] for p in state['peers']}
            promo['workers']=[{**{k:v[k] for k in ('sandbox_id','directory','owner_id','boot_id','runtime')},
                **({'preserve_role':preserve[v['sandbox_id']]} if v['sandbox_id'] in preserve else {})} for v in workers.values()]
            result=run_coreweave_promotion(directory/'promotion',promo)
            if result['phase'] not in ('workers-released','released'):raise ValueError('coordinator did not release worker authority')
            state.update(phase='primary-ready',promotion=result);write_json_durable(journal,state)
        if state['phase']=='primary-ready':
            custody=start_restored_workers(directory/'promotion',workers,state['pins'],nonce=nonce,coordinator_role=promotion['primary_role'],cwctl=cwctl,fences=fences)
            state.update(phase='complete',owner_journal=custody);write_json_durable(journal,state)
        if state['phase']!='complete':raise ValueError('unrecognized restore phase')
        return state


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--config-sha256',required=True);parser.add_argument('--journal',required=True)
    args=parser.parse_args();raw=Path(args.config).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.config_sha256:raise ValueError('restore controller config digest differs')
    print(json.dumps(run_full_restore(args.journal,json.loads(raw)),sort_keys=True))


if __name__=='__main__':main()
