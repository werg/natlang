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
        script=('import hashlib,json,os,pathlib,sys,tarfile,uuid\n'
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
                'if not store.exists():install_seed(d/"seed",store,expected_receipt_sha256=sha)\n'
                'freeze={"schema":"natlang.diloco-replica-file-freeze/1","nonce":json.loads((d/"seed"/"seed-receipt.json").read_text())["freeze_nonce"]};fp=store/"replica-files-frozen.json"\n'
                'if fp.exists() and json.loads(fp.read_text())!=freeze:raise ValueError("file custody freeze differs")\n'
                'if not fp.exists():write_json_durable(fp,freeze)\n'
                'v=json.loads((store/"seed-install.json").read_text())\n'
                'if v.get("seed_receipt_sha256")!=sha:raise ValueError("installed seed differs")\n'
                'print(json.dumps(v))\n')
        extraction=root+'/extract-'+archive_sha;store=host['store']
        if not store.startswith('/work/') or '..' in Path(store).parts:raise ValueError('invalid replica store')
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
        manifest=json.loads((staging/'staging.json').read_text());verified=json.loads((staging/'replica-restored'/'verified-restore.json').read_text())
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
            pins=stage_worker_checkpoints(staging,workers,cwctl=cwctl);state.update(phase='workers-staged',pins=pins);write_json_durable(journal,state)
        if state['phase']=='workers-staged':
            seed=directory/'seed';seed_store=directory/'seed-store';closure=staging/'diloco-replica-recovery'
            if not seed.exists():
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
