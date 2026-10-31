import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.neuralese_data import provider_campaign_source as source
from scripts.neuralese_data.teacher_outcome_contract import exact_honest_root_block


def ref(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes':path.stat().st_size}


def fixture(tmp_path, monkeypatch, mutation=None):
    def put(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value)+'\n');return p
    program={'id':'task','version':'natlang.program/2','split':'train','source_groups':['world-a','world-b']}
    if mutation=='honest-root-block':
        program['semantics']={'operation':'blocked'}
        program['curriculum']={'reference':{'root':[['return_result',
            {'status':'blocked','reason':'the records do not settle this'}]]}}
    cases=put('cases.jsonl',program)
    row={'id':'execution','version':'natlang.teacher_trajectory.native/1',
         'task':{'program_ir':program},
         'provenance':{'provider':'openai-codex','model':'gpt-6-luna',
                       'program_ir_sha256':source.canonical_json_sha256_hex(program)},
         'execution_identity':{'execution_run_id':'actual-episode'},
         'trajectory':[{'phase':'action','assistant':{'content':'return 1;'}}],
         'outcome':{'status':'done','accepted':True,'oracle':{'accepted':True},'checks':{'exact':True}}}
    if mutation=='honest-root-block':
        row['outcome']={'status':'quiesced','accepted':True,'oracle':{'accepted':True,'level':'exact'},
            'checks':{'seeded_failure_requirement':True,'expected_status':True,'honest_stop':True,
                'effects':True,'files':True,'world':True,'file_return_consistency':True,'answer':True},
            'invocation_ledger':[{'invocation_id':'root','parent_invocation_id':None,
                'completion_status':'quiesced','completion_source':'execution_graph'}],
            'action_ledger':[{'call_id':'root','seq':8,'name':'return_result',
                'arguments':{'status':'blocked','reason':'only the actual record is not enough'},
                'outcome':'blocked'}]}
        row['trajectory']=[{'phase':'action','invocation_id':'root','assistant':{'calls':[
            {'tool':'return_result','source_tool':'return_result',
             'arguments':{'status':'blocked','reason':'only the actual record is not enough'}}]}}]
    if mutation=='actor':row['provenance']['model']='a-student'
    if mutation=='provider':row['provenance']['provider']='another-provider'
    if mutation=='oracle':row['outcome']['oracle']['accepted']=False
    if mutation=='trajectory':row['trajectory']=[]
    result=put('result.jsonl',row);combined=tmp_path/'combined.jsonl';combined.write_bytes(result.read_bytes())
    manifest=put('result.jsonl.manifest.json',{'version':'natlang.teacher_batch.native/1',
        'source_sha256':ref(cases)['sha256'],'range':{'start':0,'count':1},'completed':1,
        'missing':[],'output_sha256':ref(result)['sha256'],'model':'gpt-6-luna'})
    queue=put('queue.jsonl',{'index':0,'output':str(result)})
    plan_path=put('plan.json',{'fixture':'validated separately by dispatcher tests'})
    plan={'source':str(cases),'source_sha256':ref(cases)['sha256'],
          'model_id':'gpt-6-luna','provider':'openai-codex','runtime_receipt_sha256':'frozen-runtime'}
    identity={'campaign_id':'campaign','source':str(cases),'runtime':str(tmp_path/'runtime'),
              'campaign_root':str(tmp_path),'slots':6,'cases':[{'index':0,'seed':7}]}
    identity['identity_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    monkeypatch.setattr(source,'validated_dispatch_plan',lambda *args:(plan,identity))
    claim={'event':'claim','claim_id':'campaign-case-000000','index':0,'slot':1,
           'source':str(cases),'runtime':identity['runtime'],'queue':str(queue),
           'queue_sha256':ref(queue)['sha256'],'identity_sha256':identity['identity_sha256']}
    terminal={**claim,'event':'terminal','exit_code':0,
              'output_accounting':{'complete':True,'output_sha256':ref(result)['sha256']}}
    if mutation=='unclosed':terminal['exit_code']=1
    events=[{'event':'campaign_open','identity':identity},claim,terminal]
    ledger=tmp_path/'ledger.jsonl';ledger.write_text(''.join(json.dumps(v)+'\n'for v in events))
    declaration={'schema':source.SCHEMA,'training_admission':False,'source':ref(combined),
        'plan':ref(plan_path),'claim_ledger':ref(ledger),'results':[{'claim_id':claim['claim_id'],
        'queue':ref(queue),'result':ref(result),'manifest':ref(manifest)}]}
    witness=put('provider-source.json',declaration)
    if mutation=='changed-output':result.write_text('{}\n')
    protection={'held':{'world-b'}if mutation=='protected-related-world'else set(),
                'quality':set(),'pending':set(),'proof':{'sha256':'protected-projection'}}
    args=SimpleNamespace(source=combined,provider_source_manifest=witness,
                         protection_manifest=tmp_path/'protected.json')
    return args,protection


def test_closed_provider_source_keeps_every_world_and_actual_actor(tmp_path,monkeypatch):
    args,protection=fixture(tmp_path,monkeypatch)
    result=source.source_binding(args,protection_reader=lambda _:protection)
    assert result['train_groups']=={'world-a','world-b'}
    assert result['teacher']['teacher']=='gpt-6-luna'
    assert result['transfer']['training_admission']is False
    assert result['collections'][next(iter(result['collections']))]['runtime_model']=='gpt-6-luna'


def test_provider_source_accepts_exact_honest_root_block(tmp_path,monkeypatch):
    args,protection=fixture(tmp_path,monkeypatch,'honest-root-block')
    result=source.source_binding(args,protection_reader=lambda _:protection)
    assert result['selected_ids']=={'task'}


@pytest.mark.parametrize('mutation',['actor','provider','oracle','trajectory','unclosed',
                                     'changed-output','protected-related-world'])
def test_provider_source_rejects_false_success_actor_or_custody(tmp_path,monkeypatch,mutation):
    args,protection=fixture(tmp_path,monkeypatch,mutation)
    with pytest.raises(ValueError):
        source.source_binding(args,protection_reader=lambda _:protection)


def _blocked_root_fixture():
    program = {'id': 'blocked-task', 'version': 'natlang.program/2', 'split': 'train',
        'source_groups': ['world-a'], 'semantics': {'operation': 'blocked'}, 'curriculum': {'reference': {'root': [
            ['eval', {'code': 'assess(records)'}],
            ['return_result', {'status': 'blocked', 'reason': 'records are insufficient'}],
        ]}}}
    root_id, child_id = 'root-invocation', 'child-invocation'
    row = {'id': 'blocked-execution', 'task': {'program_ir': program},
        'outcome': {'status': 'quiesced', 'accepted': True,
            'oracle': {'accepted': True, 'level': 'exact'},
            'checks': {'seeded_failure_requirement': True, 'expected_status': True, 'honest_stop': True,
                'effects': True, 'files': True, 'world': True, 'file_return_consistency': True, 'answer': True},
            'invocation_ledger': [
                {'invocation_id': root_id, 'parent_invocation_id': None,
                 'completion_status': 'quiesced', 'completion_source': 'execution_graph'},
                {'invocation_id': child_id, 'parent_invocation_id': root_id,
                 'completion_status': 'done', 'completion_source': 'execution_graph'},
            ],
            'action_ledger': [
                {'call_id': root_id, 'seq': 8, 'name': 'return_result',
                 'arguments': {'status': 'blocked', 'reason': 'The records do not settle the question.'},
                 'outcome': 'blocked'},
            ]},
        'trajectory': [
            {'invocation_id': root_id, 'assistant': {'calls': [
                {'tool': 'eval', 'source_tool': 'eval', 'arguments': {'code': 'assess(records)'}},
            ]}},
            {'invocation_id': child_id, 'assistant': {'calls': [
                {'tool': 'return_result', 'source_tool': 'return_result',
                 'arguments': {'status': 'success', 'value': {'enough': False}}},
            ]}},
            {'invocation_id': root_id, 'assistant': {'calls': [
                {'tool': 'return_result', 'source_tool': 'return_result',
                 'arguments': {'status': 'blocked', 'reason': 'The records do not settle the question.'}},
            ]}},
        ]}
    return row, program


def test_only_expected_honest_root_block_has_blocked_admission_exception():
    row, program = _blocked_root_fixture()
    assert exact_honest_root_block(row, program)
    child_blocked = copy.deepcopy(row)
    child_blocked['trajectory'][1]['assistant']['calls'][0]['arguments']['status'] = 'blocked'
    child_blocked['outcome']['invocation_ledger'][1]['completion_status'] = 'quiesced'
    assert not exact_honest_root_block(child_blocked, program)


    unexpected = copy.deepcopy(program)
    unexpected['curriculum']['reference']['root'][-1][1]['status'] = 'success'
    assert not exact_honest_root_block(row, unexpected)
    false_check = copy.deepcopy(row)
    false_check['outcome']['checks']['honest_stop'] = False
    assert not exact_honest_root_block(false_check, program)
    missing_terminal = copy.deepcopy(row)
    missing_terminal['trajectory'][-1]['assistant']['calls'][0]['arguments']['reason'] = ''
    assert not exact_honest_root_block(missing_terminal, program)
    missing_custody = copy.deepcopy(row)
    missing_custody['outcome']['action_ledger'] = []
    assert not exact_honest_root_block(missing_custody, program)


def test_blocked_reference_cannot_override_the_actual_task_operation():
    row, program = _blocked_root_fixture()
    program['semantics']['operation'] = 'exact'
    assert not exact_honest_root_block(row, program)


def test_expected_root_block_requires_explicit_parentless_capture():
    row, program = _blocked_root_fixture()
    del row['outcome']['invocation_ledger'][0]['parent_invocation_id']
    assert not exact_honest_root_block(row, program)


def test_historical_dispatch_attestation_uses_local_digest_override(tmp_path, monkeypatch):
    import subprocess
    from scripts import run_reviewed_luna_dispatcher as dispatcher

    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    git_path = 'scripts/run_reviewed_luna_dispatcher.py'
    historical = subprocess.run(['git', '-C', str(repo), 'show', f'{commit}:{git_path}'],
                                capture_output=True, check=True).stdout
    historical_sha = hashlib.sha256(historical).hexdigest()
    target = str((repo / git_path).resolve())
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps({'artifact_hashes': {target: historical_sha}}) + '\n')
    attestation_path = tmp_path / 'dispatch-plan-history-attestation.json'
    attestation_path.write_text(json.dumps({
        'schema': 'natlang.reviewed-dispatch-history-attestation/1',
        'plan_path': str(plan_path.resolve()),
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'git_commit': commit,
        'historical_artifacts': {target: {'git_path': git_path,
            'sha256': historical_sha, 'git_commit': commit}},
    }) + '\n')
    calls = []

    def validator(path, expected, *, digest_fn=None):
        calls.append(digest_fn)
        if len(calls) == 1:
            raise ValueError(f'Missing or changed reviewed artifact pin: {target}')
        assert digest_fn(target) == historical_sha
        assert digest_fn(plan_path) == hashlib.sha256(plan_path.read_bytes()).hexdigest()
        return {'campaign_id': 'test'}, {'identity_sha256': 'fixture'}

    monkeypatch.setattr(dispatcher, 'validate_plan', validator)
    original_digest = dispatcher.digest
    _, identity = source.validated_dispatch_plan(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())
    assert identity['dispatch_plan_history_attestation'] == str(attestation_path.resolve())
    assert dispatcher.digest is original_digest
    assert calls[0] is None and callable(calls[1])


def test_text_emulation_runtime_requires_compiled_cli_and_effective_option(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.run_reviewed_luna_dispatcher import _validate_text_neuralese_emulation

    runtime = tmp_path / 'runtime'
    (runtime / 'dist/teacher').mkdir(parents=True)
    bindings = {}
    for rel, body in [('teacher/cli.js', b'cli'), ('teacher/collector.js', b'collector')]:
        target = runtime / 'dist' / rel
        target.write_bytes(body)
        bindings[rel] = hashlib.sha256(body).hexdigest()
    (runtime / 'frozen-runtime.json').write_text('{}\n')
    manifest = {'build': {'compiled_dist_files': bindings}}
    monkeypatch.setattr('scripts.run_reviewed_luna_dispatcher.subprocess.run',
        lambda *args, **kwargs: SimpleNamespace(returncode=0,
            stdout='Options: --text-neuralese-emulation\n', stderr=''))
    proof = _validate_text_neuralese_emulation(runtime, manifest)
    assert proof['option'] == '--text-neuralese-emulation'
    monkeypatch.setattr('scripts.run_reviewed_luna_dispatcher.subprocess.run',
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='Options: --prompt-rendering\n', stderr=''))
    with pytest.raises(ValueError, match='does not implement'):
        _validate_text_neuralese_emulation(runtime, manifest)


def test_historical_dispatch_attestation_rejects_path_outside_repo(tmp_path, monkeypatch):
    from scripts import run_reviewed_luna_dispatcher as dispatcher

    repo = Path(__file__).resolve().parents[2]
    target = str((tmp_path / 'outside.py').resolve())
    pin = 'a' * 64
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps({'artifact_hashes': {target: pin}}) + '\n')
    attestation_path = tmp_path / 'dispatch-plan-history-attestation.json'
    attestation_path.write_text(json.dumps({
        'schema': 'natlang.reviewed-dispatch-history-attestation/1',
        'plan_path': str(plan_path.resolve()),
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'git_commit': '1' * 40,
        'historical_artifacts': {target: {'git_path': '../outside.py',
            'sha256': pin, 'git_commit': '1' * 40}},
    }) + '\n')
    monkeypatch.setattr(dispatcher, 'validate_plan',
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError(f'Missing or changed reviewed artifact pin: {target}')))
    with pytest.raises(ValueError, match='outside the repository'):
        source.validated_dispatch_plan(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())


def test_historical_dispatch_attestation_rejects_wrong_git_object_hash(tmp_path, monkeypatch):
    import subprocess
    from scripts import run_reviewed_luna_dispatcher as dispatcher

    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True).stdout.strip()
    git_path = 'scripts/run_reviewed_luna_dispatcher.py'
    target = str((repo / git_path).resolve())
    pin = 'f' * 64
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps({'artifact_hashes': {target: pin}}) + '\n')
    attestation_path = tmp_path / 'dispatch-plan-history-attestation.json'
    attestation_path.write_text(json.dumps({
        'schema': 'natlang.reviewed-dispatch-history-attestation/1',
        'plan_path': str(plan_path.resolve()),
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'git_commit': commit,
        'historical_artifacts': {target: {'git_path': git_path,
            'sha256': pin, 'git_commit': commit}},
    }) + '\n')
    monkeypatch.setattr(dispatcher, 'validate_plan',
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError(f'Missing or changed reviewed artifact pin: {target}')))
    with pytest.raises(ValueError, match='differs from its Git object'):
        source.validated_dispatch_plan(plan_path, hashlib.sha256(plan_path.read_bytes()).hexdigest())
