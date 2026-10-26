"""Frozen source TaskEval baseline plumbing; no model-quality claim is implied."""
import copy
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.eval.native_state import task_execution_receipt
from natlang_neuralese.eval.task_cadence import (_task_eval_runtime_files,task_eval_policy_identity,
    task_eval_runtime_identity)
from natlang_neuralese.rl.native_role_release import _validate_resolved_source_task_baselines, binding
from test_native_task_execution_receipt import IDENTITY, score_fixture


def fixture(tmp_path, successes=0):
    tmp_path.mkdir(parents=True,exist_ok=True)
    packet=tmp_path/'packet.json';packet.write_text('{}\n')
    declaration={'packet':{'path':str(packet.resolve()),'sha256':sha256_file_hex(packet)},
        'cases':[0,3,6],'reference_successes':None,'min_successes':0,'prompt_rendering':'full'}
    summary,score,score_path=score_fixture(tmp_path)
    summary['packet_sha256']=declaration['packet']['sha256']
    score['packet_sha256']=declaration['packet']['sha256']
    if successes==0:
        score['cases_detail'][0]['outcome']='invalid_action'
        score['outcome_counts']['task_success']-=1
        score['outcome_counts']['invalid_action']+=1
        score['task_successes']=0
        summary={key:value for key,value in score.items() if key!='cases_detail'}
    score_path.write_text(json.dumps(score))
    receipt=task_execution_receipt(summary,IDENTITY,checkpoint_sha256=IDENTITY['checkpoint_sha256'])
    runtime=task_eval_runtime_identity(declaration)
    receipt.update(source_baseline=True,baseline_comparison_floor=0,
        task_eval_policy_sha256=task_eval_policy_identity(declaration),task_runtime_identity=runtime)
    receipt_path=tmp_path/'source-task-receipt.json';receipt_path.write_text(json.dumps(receipt))
    score_ref={'path':str(score_path.resolve()),'sha256':sha256_file_hex(score_path)}
    receipt_ref={'path':str(receipt_path.resolve()),'sha256':sha256_file_hex(receipt_path)}
    baseline={'native_identity':IDENTITY,'task_eval_policy_sha256':receipt['task_eval_policy_sha256'],
        'task_runtime_identity':runtime,'task_successes':successes,
        'task_execution_receipt':receipt_ref,'score':score_ref}
    provenance={'kind':'native-source-task-baseline/1','source_identity':IDENTITY,
        'task_eval_policy_sha256':receipt['task_eval_policy_sha256'],
        'task_execution_receipt':receipt_ref,'task_runtime_identity':runtime,'score':score_ref,
        'superseded_declared_reference':{'reference_successes':None,'reference_provenance':None}}
    source={'models':{'lfm:base':{'task_eval':declaration}},'task_eval':declaration}
    effective=copy.deepcopy(source);effective['snapshot_preparation']=False
    effective['models']['lfm:base']['task_eval']['reference_successes']=successes
    effective['models']['lfm:base']['task_eval']['reference_provenance']=provenance
    # The model-specific declaration is what this participant actually uses.
    source_path=tmp_path/'source-curriculum.json';source_path.write_text(json.dumps(source))
    proof={'curriculum':binding(source_path),'snapshot_preparation':{'participants':{'lfm:base':IDENTITY}},
        'source_task_baselines':{'lfm:base':baseline}}
    return proof,effective


def test_task_policy_identity_excludes_only_the_measured_comparison_baseline(tmp_path):
    packet=tmp_path/'packet.json';packet.write_text('{}')
    base={'packet':{'path':str(packet),'sha256':sha256_file_hex(packet)},'cases':[0],
        'reference_successes':None,'min_successes':0}
    resolved={**base,'reference_successes':0,'reference_provenance':{'measured':'receipt'}}
    assert task_eval_policy_identity(base)==task_eval_policy_identity(resolved)
    assert task_eval_policy_identity(base)!=task_eval_policy_identity({**base,'cases':[1]})


def test_task_runtime_inventory_tracks_imported_typescript_and_python_files(tmp_path):
    repo=tmp_path/'repo';runtime=repo/'ts-host'
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    put(runtime/'dist/teacher/cli.js','export * from "./agent.js";')
    put(runtime/'dist/teacher/agent.js','export const value = 1;')
    put(runtime/'prelude.js','')
    put(runtime/'package.json','{}')
    put(repo/'package.json','{}');put(repo/'package-lock.json','{}')
    put(repo/'training/source-reviews/holds.jsonl','')
    put(repo/'training/source-reviews/family-holds.json','{}')
    put(repo/'training/source-reviews/datasets.json','{}')
    put(repo/'scripts/run_bonsai_queue.py','print("collector")')
    package=repo/'training/neuralese/natlang_neuralese'
    put(package/'serve/engine.py','VALUE = 1')
    first=_task_eval_runtime_files(repo,runtime)
    first_by_path={row['path']:row for row in first}
    ts_agent=str((runtime/'dist/teacher/agent.js').resolve())
    cli=str((runtime/'dist/teacher/cli.js').resolve())
    engine=str((package/'serve/engine.py').resolve())
    put(runtime/'dist/teacher/agent.js','export const value = 2;')
    second=_task_eval_runtime_files(repo,runtime)
    second_by_path={row['path']:row for row in second}
    assert first_by_path[cli]['sha256']==second_by_path[cli]['sha256']
    assert first_by_path[ts_agent]['sha256']!=second_by_path[ts_agent]['sha256']
    assert set(second_by_path)==set(first_by_path)
    put(runtime/'dist/teacher/agent.js','export const value = 1;')
    put(package/'serve/engine.py','VALUE = 2')
    third_by_path={row['path']:row for row in _task_eval_runtime_files(repo,runtime)}
    assert third_by_path[cli]['sha256']==first_by_path[cli]['sha256']
    assert third_by_path[engine]['sha256']!=first_by_path[engine]['sha256']


def test_zero_is_a_measured_source_baseline_and_binds_effective_policy(tmp_path):
    proof,effective=fixture(tmp_path,successes=0)
    assert _validate_resolved_source_task_baselines(proof,effective,['lfm:base'])


def test_source_baseline_rejects_policy_or_weight_substitution(tmp_path):
    proof,effective=fixture(tmp_path,successes=0)
    changed=copy.deepcopy(effective);changed['models']['lfm:base']['task_eval']['cases']=[1]
    with pytest.raises(ValueError,match='execution policy'):
        _validate_resolved_source_task_baselines(proof,changed,['lfm:base'])
    proof,effective=fixture(tmp_path/'second',successes=0)
    proof['snapshot_preparation']['participants']['lfm:base']=copy.deepcopy(
        proof['snapshot_preparation']['participants']['lfm:base'])
    proof['snapshot_preparation']['participants']['lfm:base']['checkpoint_sha256']='d'*64
    with pytest.raises(ValueError,match='source checkpoint'):
        _validate_resolved_source_task_baselines(proof,effective,['lfm:base'])


def test_source_baseline_binds_runtime_files_and_commit(tmp_path):
    proof,effective=fixture(tmp_path,successes=0)
    proof=copy.deepcopy(proof)
    baseline=proof['source_task_baselines']['lfm:base']
    baseline['task_runtime_identity']['files'][0]['sha256']='e'*64
    with pytest.raises(ValueError,match='runtime|policy|mismatch'):
        _validate_resolved_source_task_baselines(proof,effective,['lfm:base'])


def test_completion_dependency_closure_includes_baseline_runtime_and_scores(tmp_path):
    proof,effective=fixture(tmp_path,successes=0)
    source_path=Path(proof['curriculum']['path']);source=json.loads(source_path.read_text())
    pairs=tmp_path/'pairs.jsonl';pairs.write_text('{}\n')
    held=tmp_path/'held.jsonl';held.write_text('{}\n')
    source.update(pairs=binding(pairs),held_pairs=binding(held));source_path.write_text(json.dumps(source))
    proof['curriculum']=binding(source_path)
    resolved_path=tmp_path/'resolved-curriculum.json';effective.update(pairs=binding(pairs),held_pairs=binding(held))
    resolved_path.write_text(json.dumps(effective));proof['resolved_curriculum']=binding(resolved_path)
    checkpoint=tmp_path/'shared.pt';checkpoint.write_bytes(b'checkpoint')
    adapter=tmp_path/'adapter.json';adapter.write_text('{}')
    semantic_runtime=tmp_path/'semantic-runtime.json';semantic_runtime.write_text('{}')
    runtime_certificate=tmp_path/'runtime-cert.json';runtime_certificate.write_text('{}')
    completion={'checkpoint':binding(checkpoint),'adapter':binding(adapter),'curriculum':proof['curriculum'],
        'resolved_curriculum':proof['resolved_curriculum'],
        'semantic_runtime_certificate':binding(semantic_runtime),
        'runtime_certificates':{'lfm:base':binding(runtime_certificate)},
        'task_execution_receipts':{'lfm:base':proof['source_task_baselines']['lfm:base']['task_execution_receipt']},
        'source_task_baselines':proof['source_task_baselines']}
    completion_path=tmp_path/'shared-stage-completion.json';completion_path.write_text(json.dumps(completion))
    from natlang_neuralese.rl.native_role_release import _semantic_completion_dependencies
    dependencies={row['path']:row for row in _semantic_completion_dependencies(completion_path,completion)}
    for reference in proof['source_task_baselines']['lfm:base']['task_runtime_identity']['files']:
        assert reference['path'] in dependencies
    baseline=proof['source_task_baselines']['lfm:base']
    assert baseline['task_execution_receipt']['path'] in dependencies
    assert baseline['score']['path'] in dependencies
