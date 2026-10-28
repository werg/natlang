"""Frozen source TaskEval baseline plumbing; no model-quality claim is implied."""
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.eval.task_cadence import _task_eval_runtime_files,task_eval_policy_identity


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
