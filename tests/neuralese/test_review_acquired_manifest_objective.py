import json
import pytest
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from scripts.neuralese_data import review_acquired_manifest_objective as mod


def write(path,value):
    path.write_text(json.dumps(value));return mod.bind(path)


def test_existing_family_objective_reports_cap_and_tool_subset_without_admission(tmp_path,monkeypatch):
    family=('coding','native-communication','natlang','general-agent')
    obj={'name':'own-outputs/1','family_balance':{'key':'logical-family','normalization':'declared/1',
        'max_scale':8,'scales':dict.fromkeys(family,1.)}}
    monkeypatch.setattr(mod,'policy_objective',lambda _:obj)
    monkeypatch.setattr(mod,'identity',lambda _:{'sha256':'objective'})
    monkeypatch.setattr(mod,'predecessor_mass_census',lambda *_:({'families':{x:{'unscaled_mass':100.} for x in family},
        'unscaled_mass_total':400.},{'scope':'test'}))
    ce=write(tmp_path/'ce.json',{});pred=write(tmp_path/'pred.json',{'ce_policy':ce,'objective_transition':{'to':{'sha256':'objective'}}})
    combined=write(tmp_path/'combined.json',{'schema':'natlang.combined-acquired-screened-delta/1','training_admission':False,
        'predecessor':pred,'rows':1,'pending_render_obligations':{'rows':472},'carried_over_context_obligations':{'rows':7}})
    census=write(tmp_path/'census.json',{'schema':'natlang.acquired-effective-learning-census/1','training_admission':False,
        'screen':combined,'ce_policy':ce,'rows':1,'counts':{'zero_effective_rows':0,'tool_call_markup_effective_mass':60},
        'category_mass':{'reasoning':5,'reply':95},'objective_sha256':canonical_json_sha256_hex(obj)})
    result=mod.review(pred['path'],combined['path'],census['path'],tmp_path/'out.json',share=.07)
    assert result['training_admission'] is False and result['logical_family']=='general-agent'
    assert result['new_unscaled_mass']==100 and result['new_tool_markup_subset_mass']==60
    assert result['proposed_family_balance']['unscaled_mass']['general-agent']==200
    assert result['new_tool_markup_effective_mass']==pytest.approx(60*result['proposed_family_balance']['scales']['general-agent'])
