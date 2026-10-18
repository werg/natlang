"""Actual preparation/custody tests; never collect model or protected samples."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
from scripts.coreweave import teacher_source_sequence as sequence
from scripts.coreweave import teacher_incremental_export as exporter
from natlang_neuralese.rl import task_rollouts
from test_teacher_source_sequence import setup,put


def feeder_functions():
    path=Path(sequence.__file__).with_name('teacher_collection_feeder.py')
    names={'prepare_collection_command','resumed_collection_rows'}
    tree=ast.parse(path.read_text())
    namespace={'collection_sampling':sequence.collection_sampling,
               'verify_collection_sampling':sequence.verify_collection_sampling}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names],
                            type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def annotated_packet(tmp_path,sampling):
    folder,state,config=setup(tmp_path)
    manifest_path=Path(config['source_sequence']['ready_manifest'])
    manifest=json.loads(manifest_path.read_bytes());entry=manifest['entries'][0]
    for key in ('candidate_packet','qualified_packet'):
        path=Path(entry[key]['path']);value=json.loads(path.read_bytes())
        value['collection_sampling']=copy.deepcopy(sampling);entry[key]=put(path,value)
    return folder,state,config,entry


@pytest.mark.parametrize('sampling',[{'group':4,'root_mode':'authored'},{'group':4,'root_mode':'student'}])
def test_packet_transports_exact_candidate_and_prepared_sampling_authority(tmp_path,sampling):
    _,_,config,entry=annotated_packet(tmp_path,sampling)
    value=sequence._packet(entry,config['source_sequence'])
    assert value['collection_sampling']==sampling
    assert value['collection_sampling_authority']=={
        'candidate_packet':entry['candidate_packet'],'qualified_packet':entry['qualified_packet']}


@pytest.mark.parametrize('sampling',[None,{}, {'group':True,'root_mode':'student'},
    {'group':1,'root_mode':'student'},{'group':4.0,'root_mode':'student'},
    {'group':4,'root_mode':'unknown'}, {'group':4,'root_mode':'student','extra':1}])
def test_malformed_candidate_sampling_rejects_before_source_publication(tmp_path,sampling):
    _,_,config,entry=annotated_packet(tmp_path,sampling)
    original=Path(config['source_sequence']['source_configuration']).read_bytes()
    with pytest.raises(ValueError,match='sampling'):
        sequence._packet(entry,config['source_sequence'])
    assert Path(config['source_sequence']['source_configuration']).read_bytes()==original


@pytest.mark.parametrize('mutation',['missing','wrong_mode','float_group'])
def test_candidate_and_prepared_sampling_cannot_disagree(tmp_path,mutation):
    _,_,config,entry=annotated_packet(tmp_path,{'group':4,'root_mode':'authored'})
    path=Path(entry['qualified_packet']['path']);prepared=json.loads(path.read_bytes())
    if mutation=='missing':prepared.pop('collection_sampling')
    elif mutation=='wrong_mode':prepared['collection_sampling']['root_mode']='student'
    else:prepared['collection_sampling']['group']=4.0
    entry['qualified_packet']=put(path,prepared)
    with pytest.raises(ValueError,match='sampling'):
        sequence._packet(entry,config['source_sequence'])


def prepared_window(tmp_path,mode):
    cases=[{'id':f'train-{i}','version':'natlang.program/2','split':'train','source_groups':[f'group-{i}'],
        'curriculum':{'reference':{'root':[['eval',{'code':f'return {i};'}]]},
            'trajectory_contract':{'authored_root_eval':True,'authored_root_sha256':'original',
                                   'min_child_invocations':0}}}for i in range(12)]
    source=tmp_path/'source-cases.jsonl';source.write_text(''.join(json.dumps(row)+'\n'for row in cases))
    sampling={'group':4,'root_mode':mode}
    conf={'cases':str(source),'cases_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'rows':12,'collection_sampling':sampling}
    command=feeder_functions()['prepare_collection_command'](source,tmp_path,12,123000,conf)
    # Parse the real feeder's argv with the production CLI parser, then execute
    # actual prepare. No stub queue or duplicate root transform is manufactured.
    task_rollouts.main(command[3:])
    entries=[json.loads(line)for line in(tmp_path/'queue.jsonl').read_text().splitlines()]
    for row in entries:row.update(context_tokens=65536,text_neuralese_emulation=True)
    teacher={'source_cases':str(source),'source_cases_sha256':conf['cases_sha256'],
        'source_provenance':conf,'collection_sampling':sampling,'seed_base':123000,
        'temperature':.6,'root_mode':mode}
    context={name:(tmp_path/name).read_bytes()for name in('source-cases.jsonl','packet.jsonl','preparation.json')}
    context['teacher-source.json']=json.dumps(teacher).encode()
    return conf,teacher,entries,context


@pytest.mark.parametrize('mode',['authored','student'])
def test_actual_twelve_case_four_sample_prepare_and_publisher_root_contract(tmp_path,mode):
    conf,teacher,entries,context=prepared_window(tmp_path,mode)
    assert len(entries)==48
    sequence.verify_collection_sampling(teacher,entries,conf)
    assert feeder_functions()['resumed_collection_rows'](entries,teacher)==12
    assert exporter._execution_source(context,entries)==(12,hashlib.sha256(context['packet.jsonl']).hexdigest())
    packet=[json.loads(line)for line in context['packet.jsonl'].splitlines()]
    for row in packet:
        assert bool(row['curriculum']['trajectory_contract'].get('authored_root_eval'))==(mode=='authored')
    assert len({(row['index'],row['sample'])for row in entries})==48


@pytest.mark.parametrize('mutation',['missing_sample','duplicate_sample','seed','mode','count','private_root',
                                     'missing_preparation','source_declaration','source_mode','duplicate_key'])
def test_explicit_window_rejects_changed_queue_and_root_contract(tmp_path,mutation):
    conf,teacher,entries,context=prepared_window(tmp_path,'authored')
    if mutation=='missing_sample':entries.pop()
    elif mutation=='duplicate_sample':entries[-1]=copy.deepcopy(entries[0])
    elif mutation=='seed':entries[0]['seed']+=1
    elif mutation=='count':entries[0]['count']=2
    elif mutation=='duplicate_key':entries[-1]['key']=entries[0]['key']
    elif mutation=='mode':
        prep=json.loads(context['preparation.json']);prep['root_mode']='student'
        context['preparation.json']=json.dumps(prep).encode()
    elif mutation=='private_root':
        rows=[json.loads(line)for line in context['packet.jsonl'].splitlines()]
        rows[0]['curriculum']['trajectory_contract']['authored_root_eval']=False
        context['packet.jsonl']=''.join(json.dumps(row)+'\n'for row in rows).encode()
        prep=json.loads(context['preparation.json']);prep['packet_sha256']=hashlib.sha256(context['packet.jsonl']).hexdigest()
        context['preparation.json']=json.dumps(prep).encode()
    elif mutation=='missing_preparation':context.pop('preparation.json')
    elif mutation=='source_declaration':teacher['collection_sampling']={'group':2,'root_mode':'student'}
    elif mutation=='source_mode':teacher['root_mode']='student'
    context['teacher-source.json']=json.dumps(teacher).encode()
    with pytest.raises(ValueError):exporter._execution_source(context,entries)


def test_legacy_sampler_command_and_resume_preserve_two_student_samples():
    functions=feeder_functions()
    expected=['python','-m','natlang_neuralese.rl.task_rollouts','prepare','--cases','cases','--out','out',
        '--rows','12','--group','2','--temperature','0.6','--case-seconds','7200','--max-model-requests','128',
        '--max-turns','160','--seed-base','99','--root-mode','student']
    assert functions['prepare_collection_command']('cases','out',12,99,{})==expected
    assert functions['resumed_collection_rows']([{}]*24,{'root_mode':'student','source_provenance':{}})==12
    with pytest.raises(ValueError,match='partial sample group'):
        functions['resumed_collection_rows']([{}]*23,{'source_provenance':{}})


def test_legacy_source_configuration_has_no_new_sampling_fields(tmp_path):
    _,_,config=setup(tmp_path)
    manifest=json.loads(Path(config['source_sequence']['ready_manifest']).read_bytes())
    value=sequence._packet(manifest['entries'][0],config['source_sequence'])
    assert 'collection_sampling'not in value and 'collection_sampling_authority'not in value


@pytest.mark.parametrize('mode',['authored','student'])
def test_new_window_reservation_does_not_relabel_live_legacy_slot(tmp_path,monkeypatch,mode):
    from test_teacher_source_sequence import seal
    from test_teacher_primary_overlap import close_predecessor
    folder,state,config=setup(tmp_path)
    decl=config['source_sequence'];ready=Path(decl['ready_manifest']);manifest=json.loads(ready.read_bytes())
    entry=manifest['entries'][1];sampling={'group':4,'root_mode':mode}
    for key in ('candidate_packet','qualified_packet'):
        path=Path(entry[key]['path']);value=json.loads(path.read_bytes())
        value['collection_sampling']=sampling;entry[key]=put(path,value)
    inventory=Path(decl['candidate_inventory']['path'])
    decl['candidate_inventory']=put(inventory,{'packets':[e['candidate_packet']for e in manifest['entries']]})
    manifest['candidate_inventory']=decl['candidate_inventory'];put(ready,manifest)
    sequence.prepare_source_sequence(folder,state,config,22)
    sealed=seal(folder,state,config,22);sequence.observe_source_sequence(folder,state,config)
    active_source=Path(decl['source_configuration']);old=active_source.read_bytes()
    old_seals=copy.deepcopy(state['source_sequence']['started'])
    state.update(owner={'pid':123,'start_ticks':'4','boot_id':'boot'},active_batch=22)
    monkeypatch.setattr(sequence,'alive',lambda owner:bool(owner))
    config['primary_parallel_batches']=2
    reserved=sequence.prepare_prefetched_source(folder,state,config,23)
    intent=sequence._json(reserved['intent'],decl)
    configuration=sequence._json(intent['new_configuration'],decl)
    assert configuration['collection_sampling']==sampling
    assert active_source.read_bytes()==old
    assert state['source_sequence']['started']==old_seals
    # A restart recovers the very same reservation, not another 4-sample packet.
    assert sequence.prepare_prefetched_source(folder,state,config,23)==reserved
    root=Path(config['root'])/'evidence-collections-v2/batch-0023';root.mkdir()
    (root/'source-cases.jsonl').write_bytes(Path(configuration['cases']).read_bytes())
    entries=task_rollouts.build_queue(root/'packet.jsonl',1,4,root,temperature=.6,
        case_seconds=7200,max_model_requests=128,max_turns=160,seed_base=123000)
    for row in entries:row['text_neuralese_emulation']=True
    source={'batch':23,'source_cases_sha256':configuration['cases_sha256'],
        'actual_task_runtime':config['task_runtime'],'collection_sampling':sampling,
        'root_mode':mode,'seed_base':123000,'temperature':.6,'source_provenance':configuration}
    (root/'queue.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in entries))
    put(root/'teacher-source.json',source)
    sequence.observe_prefetched_source(folder,state,config)
    assert state['source_sequence']['started']['22']==old_seals['22']
    close_predecessor(folder,state,config,sealed)
    sequence.adopt_prefetched_source(folder,state,config,23)
    assert json.loads(active_source.read_bytes())==configuration
    assert state['source_sequence']['pending']==reserved
    assert sequence.adopt_prefetched_source(folder,state,config,23)==reserved


def test_started_sampling_receipt_rejects_a_two_sample_queue_for_four_sample_intent(tmp_path):
    folder,state,config,entry=annotated_packet(tmp_path,{'group':4,'root_mode':'authored'})
    configuration=sequence._packet(entry,config['source_sequence'])
    root=tmp_path/'started';root.mkdir()
    source={'batch':23,'source_cases_sha256':configuration['cases_sha256'],
        'actual_task_runtime':config['task_runtime'],'collection_sampling':configuration['collection_sampling'],
        'root_mode':'authored','seed_base':0,'temperature':.6,'source_provenance':configuration}
    (root/'source-cases.jsonl').write_bytes(Path(configuration['cases']).read_bytes())
    (root/'queue.jsonl').write_text('{}\n{}\n');put(root/'teacher-source.json',source)
    intent=put(tmp_path/'intent.json',{'new_configuration':put(tmp_path/'new.json',configuration),
                                    'prefetch':{'task_runtime':config['task_runtime']}})
    pending={'batch':23,'ordinal':0,'cases_sha256':configuration['cases_sha256'],'intent':intent}
    with pytest.raises(ValueError,match='sample count'):
        sequence._prefetch_start(folder,{},config['source_sequence'],pending,root)
    assert not(folder/'source-sequence/started/batch-0023.json').exists()


@pytest.mark.parametrize('mode',['authored','student'])
def test_export_validates_full_window_while_publishing_a_partial_tranche(tmp_path,mode):
    _,_,entries,context=prepared_window(tmp_path,mode)
    context['queue.jsonl']=''.join(json.dumps(row)+'\n'for row in entries).encode()
    tranche=[{**entries[7],'finish_event':{'event':'finish','key':entries[7]['key']}}]
    assert exporter._execution_source(context,tranche)[0]==12
    changed=copy.deepcopy(tranche);changed[0]['seed']+=1
    with pytest.raises(ValueError,match='differs from its original complete queue'):
        exporter._execution_source(context,changed)
    master=entries[:-1];context['queue.jsonl']=''.join(json.dumps(row)+'\n'for row in master).encode()
    with pytest.raises(ValueError,match='sample count'):
        exporter._execution_source(context,tranche)
