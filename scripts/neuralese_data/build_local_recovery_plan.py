#!/usr/bin/env python3
"""Build a draft local-recovery plan from the actual controller and full checkpoint.

This is an inventory, not a recoverability claim. Qualification of the staged
archives and an actual isolated restore must follow before a plan can run.
"""
from __future__ import annotations
import argparse,json,os,re,subprocess,time
from pathlib import Path
from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_continuation import collect_sft_progress
from natlang_neuralese.train.native_supervision import ce_policy_dependencies
from natlang_neuralese.train.recipe import native_source_dependencies,recipe_file_dependencies
from natlang_neuralese.model.native_model import _checkpoint_bound_config_inventory
from natlang_neuralese.train.checkpoint_offload import recovery_static_sft_contract


def ldd_files(path):
    result=subprocess.run(['ldd',str(path)],capture_output=True,text=True,timeout=30)
    if result.returncode and 'not a dynamic executable' not in result.stderr and 'statically linked' not in result.stderr:
        raise ValueError('cannot inventory executable dependencies: '+str(path)+' '+result.stderr[-300:])
    output=set()
    for line in result.stdout.splitlines():
        if 'not found' in line:raise ValueError('missing executable dependency: '+line.strip())
        match=re.search(r'=>\s+(/\S+)|^\s*(/\S+)',line)
        if match:output.add(str(Path(match.group(1) or match.group(2)).resolve(strict=True)))
    return output


def mapped_executables(pid):
    output=set();path=Path('/proc')/str(pid)/'maps'
    for line in path.read_text().splitlines():
        parts=line.split()
        if len(parts)>=6 and 'x' in parts[1] and parts[-1].startswith('/'):
            value=Path(parts[-1])
            if value.is_file():output.add(str(value.resolve()))
    return output


def controller_declared_inputs(config):
    """Inventory future source obligations separately from earned checkpoint state."""
    result=[];seen=set()
    for key,kind in (('seed_native_graphs','source'),('foundation_native_graphs','source'),
                     ('held_native_graphs','source'),('ce_supervision_policy','ce')):
        value=config.get(key)
        if not value:continue
        if not isinstance(value,str):raise ValueError('controller source must be a path: '+key)
        path=Path(value).resolve(strict=True);item={'kind':kind,'path':str(path),'sha256':sha(path)}
        identity=(kind,item['path'],item['sha256'])
        if identity not in seen:result.append(item);seen.add(identity)
    return result


def controller_future_controls(config):
    """Return explicitly bound controls needed by a declared future SFT stage."""
    params=config.get('seed_trainer_parameters') or {}
    if not isinstance(params,dict):raise ValueError('controller seed trainer parameters must be an object')
    result=[]
    for key in ('recipe_binding','memory_qualification','sft_packed_geometry','pending_focus'):
        value=params.get(key)
        if not value:continue
        if isinstance(value,str):
            # Recipe/geometry bindings may be serialized structured arguments.
            if value.lstrip().startswith('{'):value=json.loads(value)
            else:value={'path':value}
        if not isinstance(value,dict) or not isinstance(value.get('path'),str):
            raise ValueError('controller future control lacks a path: '+key)
        result.append({'kind':key,'path':value['path'],'sha256':value.get('sha256')})
    return result


def checkpoint_base_files(base,state):
    return _checkpoint_bound_config_inventory(
        base,(state.get('model_config') or {}).get('base_inventory'),state.get('backbone_state'))


def snapshot_control(path, directory, repo):
    """Seal exact committed helper bytes under a content-addressed immutable name."""
    import hashlib,tempfile
    source=Path(path).resolve(strict=True);repo=Path(repo).resolve(strict=True)
    relative=source.relative_to(repo).as_posix()
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    committed=subprocess.check_output(['git','-C',str(repo),'show',f'{commit}:{relative}'])
    before=source.stat();data=source.read_bytes();after=source.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if identity(before)!=identity(after) or data!=committed:
        raise ValueError('recovery helper differs from its committed source: '+str(source))
    digest=hashlib.sha256(data).hexdigest();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    target=directory/(digest+'-'+source.name)
    if not target.exists():
        fd,name=tempfile.mkstemp(prefix='.control-',dir=directory)
        try:
            with os.fdopen(fd,'wb') as out:out.write(data);out.flush();os.fsync(out.fileno())
            os.chmod(name,0o400)
            try:os.link(name,target)
            except FileExistsError:pass
        finally:Path(name).unlink(missing_ok=True)
        dfd=os.open(directory,os.O_DIRECTORY);os.fsync(dfd);os.close(dfd)
    if target.is_symlink() or file_sha256_for_control(target)!=digest or target.stat().st_mode & 0o222:
        raise ValueError('recovery helper snapshot is not immutable or differs')
    return {'path':str(target),'sha256':digest,'source_path':str(source),'source_commit':commit}


def file_sha256_for_control(path):
    import hashlib
    with open(path,'rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


def historical_program_bindings(receipts,add):
    historical=[];seen=set()
    for receipt_name in receipts:
        receipt_path=Path(receipt_name).resolve(strict=True);receipt=json.loads(receipt_path.read_text())
        if receipt.get('schema')!='natlang.historical-source-program-preservation/1':
            raise ValueError('invalid historical source program preservation receipt')
        receipt_binding=add(receipt_path,'controls')
        for item in receipt.get('programs',[]):
            original=Path(item['original_path'])
            if not original.is_absolute() or str(original) in seen:raise ValueError('ambiguous historical program path')
            archive=add(item['path'],'controls',item['sha256'])
            if Path(archive['path']).stat().st_size!=item['bytes']:raise ValueError('historical program size differs')
            historical.append({'original_path':str(original),'archive':{'path':archive['path'],'sha256':archive['sha256']},
                               'receipt':{'path':receipt_binding['path'],'sha256':receipt_binding['sha256']}})
            seen.add(str(original))
    return historical


def build(args):
    config_path=Path(args.config).resolve(strict=True);config=json.loads(config_path.read_text())
    # Capture the actual process mappings before potentially slow checkpoint
    # and source reads; a short-lived trainer may exit during those reads.
    proc_stat=Path('/proc')/str(args.trainer_pid)/'stat'
    start_before=proc_stat.read_text().rsplit(')',1)[1].split()[19]
    trainer_mappings=mapped_executables(args.trainer_pid)
    if proc_stat.read_text().rsplit(')',1)[1].split()[19]!=start_before:
        raise ValueError('trainer process changed while inventorying mappings')
    process_observation={'pid':args.trainer_pid,'start_ticks':start_before,'time_unix':time.time(),
                         'scope':'point-in-time executable mappings; not a live-process lease'}
    state_path=Path(args.controller_state).resolve(strict=True);checkpoint=Path(args.checkpoint).resolve(strict=True)
    before=checkpoint.stat();state=CheckpointPolicy.load(checkpoint);after=checkpoint.stat()
    ident=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    if ident(before)!=ident(after):raise ValueError('checkpoint changed during plan inventory')
    training=state.get('training') or {};options=training.get('options') or {};receipts=training.get('source_receipts') or []
    if len(receipts)<2 or not training.get('optimizer') or not training.get('rng'):raise ValueError('plan parent is not a full-state checkpoint')
    inputs={'train':{k:receipts[0][k] for k in ('path','sha256')},'held':{k:receipts[1][k] for k in ('path','sha256')},
            'ce':options.get('ce_supervision_policy_binding')}
    foundation=options.get('foundation_native_graphs')
    inputs['foundation']=({'path':str(Path(foundation).resolve(strict=True)),'sha256':sha(Path(foundation))} if foundation else None)
    closure=[];seen=set()
    def add(path,role,expected=None,*,logical=False):
        given=Path(path)
        if not given.is_absolute():raise ValueError('recovery closure path is not absolute')
        resolved=given.resolve(strict=True)
        if not resolved.is_file():raise ValueError('recovery closure input is not a file: '+str(given))
        digest=sha(resolved)
        if expected is not None and digest!=expected:raise ValueError('recovery input digest differs: '+str(given))
        record={'role':role,'path':str(given if logical else resolved),'sha256':digest};key=(record['role'],record['path'],digest)
        if key not in seen:closure.append(record);seen.add(key)
        return record
    def add_source(ref):
        add(ref['path'],'sources',ref['sha256'])
        for item in native_source_dependencies(ref['path']):add(item['path'],'sources',item['sha256'])
    def add_ce(ref):
        if ref:
            for item in ce_policy_dependencies(ref):add(item['path'],'controls',item['sha256'])
    for role in ('train','foundation','held'):
        if inputs[role]:add_source(inputs[role])
    add_ce(inputs['ce']);additional=[];declared={(('ce' if role=='ce' else 'source'),ref['path'],ref['sha256']) for role,ref in inputs.items() if ref}
    for frame in collect_sft_progress(training,borrow_frames=True).values():
        for ref in frame.get('source_receipts') or []:
            item={'kind':'source','path':ref['path'],'sha256':ref['sha256']}
            if tuple(item[k] for k in ('kind','path','sha256')) not in declared:additional.append(item);declared.add(tuple(item[k] for k in ('kind','path','sha256')));add_source(item)
        prior=(frame.get('options') or {}).get('ce_supervision_policy_binding')
        if prior and ('ce',prior['path'],prior['sha256']) not in declared:
            item={'kind':'ce','path':prior['path'],'sha256':prior['sha256']};additional.append(item);declared.add(('ce',prior['path'],prior['sha256']));add_ce(prior)
        previous=(frame.get('options') or {}).get('foundation_native_graphs')
        if previous and not any(kind=='source' and path==previous for kind,path,_ in declared):
            path=Path(previous).resolve(strict=True);item={'kind':'source','path':str(path),'sha256':sha(path)};additional.append(item);declared.add(('source',item['path'],item['sha256']));add_source(item)
    # The controller may have qualified future stages which are not yet earned
    # in this checkpoint. Preserve their complete dependencies without crediting
    # any visits or changing the checkpoint's active source/frame identity.
    for item in controller_declared_inputs(config):
        identity=tuple(item[k] for k in ('kind','path','sha256'))
        if identity not in declared:
            additional.append(item);declared.add(identity)
            (add_ce if item['kind']=='ce' else add_source)(item)
    controller_config=Path(args.controller_config).resolve(strict=True)
    add(config_path,'controls');add(controller_config,'controls');add(args.asset_guard,'controls');add(args.asset_inventory,'controls')
    if config.get('cases'):add(config['cases'],'controls')
    if config.get('history_reasoning_binding_path'):
        add(config['history_reasoning_binding_path'],'controls',config.get('history_reasoning_binding_sha256'))
    for item in controller_future_controls(config):
        ref=add(item['path'],'controls',item['sha256'])
        if item['kind']=='recipe_binding':
            for dependency in recipe_file_dependencies(ref['path']):
                add(dependency['path'],'controls',dependency['sha256'])
    binding=options.get('recipe_binding')
    if isinstance(binding,str):binding=json.loads(binding)
    if not isinstance(binding,dict) or not binding.get('path') or not binding.get('sha256'):
        raise ValueError('checkpoint lacks its exact active recipe binding')
    recipe=Path(binding['path']).resolve(strict=True);add(recipe,'controls',binding['sha256'])
    for ref in recipe_file_dependencies(recipe):add(ref['path'],'controls',ref['sha256'])
    verifier=Path(__file__).resolve().with_name('verify_local_recovery_export.py')
    for path in args.control:
        if Path(path).resolve(strict=True)!=verifier:add(path,'controls')
    base=Path(config['base']).resolve(strict=True)
    if options.get('base') and Path(options['base']).resolve(strict=True)!=base:
        raise ValueError('controller template and checkpoint base differ')
    # A full native checkpoint supplies the actual backbone weights.  The
    # production loader validates its pinned inventory against these small
    # config/tokenizer inputs before restoring the saved backbone state.
    local_inventory=checkpoint_base_files(base,state)
    for name,digest in sorted(local_inventory.items()):
        p=base/name
        add(p,'base',digest,logical=True);add(p,'tokenizer',digest,logical=True)
    node=Path(args.node).resolve(strict=True);interpreter=Path(args.interpreter).resolve(strict=True)
    add(node,'node');add(interpreter,'interpreter')
    for path in sorted(ldd_files(node)|ldd_files(interpreter)|trainer_mappings):
        p=Path(path)
        # Runtime, venv, checkpoint and model bytes are already sealed separately.
        if any(p.is_relative_to(root) for root in (Path(args.runtime).resolve(),Path(args.venv).resolve(),base)) or p==checkpoint:continue
        add(p,'system-libraries')
    runtime=add(args.runtime_archive,'runtime');environment=add(args.environment_archive,'environment');stdlib=add(args.system_python_archive,'system-python')
    assets=add(args.asset_archive,'task-assets');assets_v3=add(args.asset_v3_archive,'task-assets')
    for path in args.archive_receipt:add(path,'controls')
    historical=historical_program_bindings(getattr(args,'historical_program_receipt',[]),add)
    repo=Path(__file__).resolve().parents[2]
    snapshots=[snapshot_control(path,Path(args.journal_directory)/'control-snapshots',repo)
               for path in (Path(__file__).resolve(),Path(args.qualifier).resolve(),verifier)]
    for snapshot in snapshots:add(snapshot['path'],'controls',snapshot['sha256'])
    config_ref={'path':str(controller_config),'sha256':sha(controller_config)}
    plan={'schema':'natlang.local-full-recovery-plan/1','session':args.session,'destination':str(Path(args.destination).resolve()),
          'journal_directory':str(Path(args.journal_directory).resolve()),'controller':{'state_path':str(state_path),'config':config_ref},
          'process_observation':process_observation,
          'inputs':inputs,'additional_inputs':additional,'closure':closure,'control_snapshot':snapshots,'historical_programs':historical,
          'require_static_contract':True,'sft_static_contract':recovery_static_sft_contract(training),
          'restore':{'runtime_archive':runtime['path'],'environment_archive':environment['path'],'system_python_archive':stdlib['path'],
                     'base_path':str(base),'node_path':str(node),'interpreter_path':str(interpreter),
                     'sample_graph':str(Path(args.sample_graph).resolve(strict=True)),'tokenizer_fingerprint':args.tokenizer_fingerprint,
                     'asset_guard':str(Path(args.asset_guard).resolve(strict=True)),'asset_inventory':str(Path(args.asset_inventory).resolve(strict=True)),
                     'asset_archives':[{'env':'NATLANG_RL_TASK_ASSETS','original_root':args.asset_root,'archive':assets['path']},
                                       {'env':'NATLANG_RL_TASK_ASSETS_V3','original_root':args.asset_v3_root,'archive':assets_v3['path']}]}}
    sample=Path(plan['restore']['sample_graph']);add(sample,'sources')
    output=Path(args.output).resolve()
    if output.exists():raise ValueError('draft recovery plan already exists')
    write_json_durable(output,plan)
    return {'path':str(output),'sha256':sha(output),'files':len(closure),'checkpoint_step':training.get('global_step'),
            'qualifier_snapshot':next(item for item in snapshots if item['source_path']==str(Path(args.qualifier).resolve(strict=True))),
            'checkpoint_observation_only':True,'status':'draft-awaiting-independent-restore-qualification'}


def main():
    p=argparse.ArgumentParser()
    for name in ('config','controller-config','controller-state','checkpoint','runtime','venv','runtime-archive','environment-archive','system-python-archive','asset-archive','asset-v3-archive','asset-root','asset-v3-root','asset-guard','asset-inventory','node','interpreter','sample-graph','tokenizer-fingerprint','destination','journal-directory','session','qualifier','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--trainer-pid',required=True,type=int);p.add_argument('--control',action='append',default=[]);p.add_argument('--archive-receipt',action='append',default=[])
    p.add_argument('--historical-program-receipt',action='append',default=[])
    a=p.parse_args();print(json.dumps(build(a)),flush=True)

if __name__=='__main__':main()
