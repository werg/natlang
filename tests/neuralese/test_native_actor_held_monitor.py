import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

path=Path(__file__).parents[2]/'scripts/neuralese_data/evaluate_native_actor.py'
spec=importlib.util.spec_from_file_location('native_actor_held',path)
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)


@pytest.mark.parametrize('failure',[None,'before','after','collector'])
def test_current_actor_monitor_is_bound_once_and_never_loads_a_model(tmp_path,monkeypatch,failure):
    calls=[];read=0
    class Server:
        def __init__(self,url,timeout):assert url=='http://same-actor'
        def get_bytes(self,path):
            nonlocal read
            if path=='/v1/models':return json.dumps({'data':[{'id':'actual-current-model'}]}).encode()
            read+=1
            sha='wrong'if failure=='before'or(failure=='after'and read>1)else'current'
            return json.dumps({'rl_policy_binding':{'checkpoint_sha256':sha},
                'native_state':{'checkpoint_sha256':sha,'enabled':True},'prompt_rendering':'full'}).encode()
    class Evaluation:
        def __init__(self,spec,out,engine,**kwargs):
            assert engine.backbone is None and engine.model_name=='actual-current-model'
            assert kwargs['external_actor']=={'url':'http://same-actor'}
        def run(self,step):
            calls.append(step)
            if failure=='collector':raise OSError('owned collector failed')
            return {'eligible':True,'cases':16,'task_successes':3}
    monkeypatch.setattr(monitor,'Server',Server);monkeypatch.setattr(monitor,'TaskEval',Evaluation)
    spec_path=tmp_path/'spec.json';spec_path.write_text(json.dumps({'packet':{'path':'protected','sha256':'bound'},'neuralese_server':True}))
    kwargs=dict(endpoint='http://same-actor',checkpoint_sha256='current',spec_path=spec_path,
        spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),out=tmp_path/'held-only',step=123)
    if failure in('before','after'):
        with pytest.raises(ValueError,match='actor'):monitor.evaluate(**kwargs)
        assert not(kwargs['out']/'completed.json').exists()
        assert calls==([]if failure=='before'else[123])
    else:
        result=monitor.evaluate(**kwargs);assert monitor.evaluate(**kwargs)==result
        assert calls==[123] and result['training_admission']is False
        assert result['status']==('monitoring-failed'if failure else'finished')
        assert result.get('eligible')is False if failure else result['summary']['eligible']is True


def test_external_task_execution_reuses_same_queues_without_engine_lifecycle(tmp_path,monkeypatch):
    from natlang_neuralese.eval import task_cadence as cadence,task_packet
    evaluator=cadence.TaskEval.__new__(cadence.TaskEval)
    evaluator.spec=cadence.parse_task_eval({'packet':{'path':'protected','sha256':'bound'},'cases':[0],
        'workers':1,'neuralese_server':True,'identical_turn_limit':3})
    evaluator.rows=[{'id':'held','split':'test'}];evaluator.subset=tmp_path/'subset.jsonl'
    evaluator.subset.write_text('{}\n');evaluator.prompt_rendering='full';evaluator.repo=tmp_path
    evaluator.engine=SimpleNamespace(backbone=None,prompt_rendering='full',model_name='current-actor')
    evaluator.external_actor={'url':'http://same-current-actor'}
    run=tmp_path/'held';run.mkdir();sentinel=run/'already-finished-journal';sentinel.write_text('preserve earned held result')
    commands=[]
    monkeypatch.setattr(cadence.subprocess,'Popen',lambda argv,**kwargs:commands.append(argv)or object())
    monkeypatch.setattr(cadence,'wait_collectors',lambda children,queues,launch,**kw: [log.close()for _,log in children]or[0])
    monkeypatch.setattr(task_packet,'score_rows',lambda rows,results,finishes,keys:{'cases':len(rows)})
    score,_=evaluator._execute(run)
    assert score=={'cases':1} and sentinel.read_text()=='preserve earned held result'
    argv=commands[0]
    assert argv[argv.index('--server')+1]=='http://same-current-actor'
    assert argv[argv.index('--model-id')+1]=='current-actor'
    entry=json.loads((run/'worker-00/queue.jsonl').read_text())
    assert(entry['case_seconds'],entry['max_model_requests'],entry['max_turns'])==(7200,128,160)
    assert '--neuralese-server'in argv and argv[argv.index('--identical-turn-limit')+1]=='3'
