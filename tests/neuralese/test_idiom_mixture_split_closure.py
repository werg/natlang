import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('build_idiom_mixture', Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/build_idiom_mixture.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def put(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_held_closes_through_ordinary_communication_and_prior(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    ordinary=tmp_path/'ordinary.jsonl'; comm=tmp_path/'comm.jsonl'
    put(ordinary,[{'split':'train','source_groups':['b','c']}, {'split':'train','source_groups':['x']}])
    put(comm,[{'split':'train','source_groups':['a','b']}, {'split':'train','source_groups':['x','y']}])
    monkeypatch.setattr(mod, 'graph_files', lambda render, split: [('program-header', ordinary)])
    config={'renders':['fixture'],'communication':[{'kind':'native-graphs','path':str(comm)}]}
    policy={'held_carve':{'fraction':1.0,'salt':'s','max_rows':10}}
    carve, held = mod.carve_groups(config,policy,held_groups={'a'},prior_components=[('c','d')])
    assert held == {'a','b','c','d'}
    assert carve == {'x','y'}
    assert mod.communication_is_held({'source_groups':['y']}, [{}], held, carve)
    assert not mod.communication_is_held({'source_groups':['z']}, [{}], held, carve)
    with pytest.raises(ValueError,match='prior train sources are connected'):
        mod.carve_groups(config,policy,trained_groups={'d'},held_groups={'a'},prior_components=[('c','d')])


def test_held_closure_applies_without_carve(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'REPO', tmp_path)
    comm=tmp_path/'comm.jsonl';put(comm,[{'split':'train','source_groups':['a','b']}])
    carve, held = mod.carve_groups({'renders':[],'communication':[{'kind':'native-graphs','path':str(comm)}]},
                                   {},held_groups={'a'})
    assert carve == set() and held == {'a','b'}
