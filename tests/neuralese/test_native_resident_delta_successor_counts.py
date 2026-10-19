import copy,json
from pathlib import Path
import pytest
from natlang_neuralese.train import native_resident as r
from natlang_neuralese.common.jsonio import write_json_durable as write

@pytest.mark.parametrize('successor',[False,True])
def test_only_successor_plan_uses_actual_cache_counts(tmp_path,monkeypatch,successor):
 from test_native_resident import setup
 monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
 paths,directory,manager,_=setup(tmp_path,manifest_train=True)
 initial=json.loads((directory/'initial.json').read_text());initial['rows']=999;write(directory/'initial.json',initial);before=(directory/'initial.json').read_bytes()
 plan=json.loads((directory.parent/'recipe-plan.json').read_text());old=copy.deepcopy(plan['recipe'])
 if successor:
  old['stages'][0]['parameters']['checkpoint_seconds']=900;plan['recipe']=old;write(directory.parent/'recipe-plan.json',plan)
 extra=tmp_path/'extra.jsonl';extra.write_text(paths['new'].read_text().splitlines()[-1]+'\n')
 wrapper=tmp_path/'union.json';write(wrapper,{'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs',**r.binding(paths['old'])},{'kind':'native-graphs',**r.binding(extra)}]})
 new=copy.deepcopy(old);new['input_bindings'].update(train=r.binding(wrapper),prior=r.binding(paths['old']));new['stages'][0]['inputs']['continued_native_graphs']='prior'
 recipe=tmp_path/'append.json';write(recipe,new)
 def prepare():return r.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),publish=False,delta_sources={'train':r.binding(extra)})
 if not successor:
  with pytest.raises(ValueError,match='counts differ'):prepare()
 else:
  result=json.loads(r.bound(prepare()).read_text())['payload'];assert result['rows']==3
  assert result['delta_validation']['prior_rows']==2
  authority=result['delta_validation']['prior_count_authority'];assert authority['kind']=='authenticated-plan-indexes'
  assert authority['plan_sha256']==r.digest(plan)and authority['historical_initial_plan_sha256']==initial['plan_sha256']
 assert(directory/'initial.json').read_bytes()==before
 assert not(directory/'pending.json').exists()
