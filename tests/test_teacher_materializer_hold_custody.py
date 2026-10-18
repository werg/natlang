import copy
import pytest
from scripts.coreweave.teacher_tranche_admission import classify_materializer_holds


def rows():
    return {'files':{'outcome':{'accepted':True,'files_check':{'failed':['item.txt'],'score':.9},'oracle':{'level':'exact','accepted':True}}},
            'agreement':{'outcome':{'accepted':True,'files_check':{'failed':[],'score':1},'oracle':{'level':'agreement','accepted':True,'score':.95}}}}


def test_current_policy_distinguishes_whole_row_holds_without_mutation():
    raw=rows();before=copy.deepcopy(raw);result=classify_materializer_holds(raw,['files','agreement'])
    assert raw==before
    assert result['files']['quality_reason']=='quality_pending_partial_files'
    assert result['agreement']['quality_reason']=='quality_pending_partial_agreement'
    assert all(len(v['policy_bindings'])==2 and 'no graph' in v['assigned_use']for v in result.values())


@pytest.mark.parametrize('fault',['full_score','pending','wrong_level','not_positive','unknown_id','duplicate_id','nonfinite','reviewed_trace'])
def test_no_unknown_hold_or_partial_score_waiver(fault):
    raw=rows();ids=['agreement'];o=raw['agreement']['outcome']
    if fault=='full_score':o['oracle']['score']=1
    elif fault=='pending':o['quality_pending']=['unreviewed']
    elif fault=='wrong_level':o['oracle']['level']='exact'
    elif fault=='not_positive':o['accepted']=False
    elif fault=='unknown_id':ids=['missing']
    elif fault=='duplicate_id':ids=['agreement','agreement']
    elif fault=='nonfinite':o['oracle']['score']=float('nan')
    else:raw['agreement']['provenance']={'trace_sha256':'63975b4cef1a6e5cf1d1d1ee02928875cd136617c97670857f2a231bbb125b75'}
    with pytest.raises(ValueError):classify_materializer_holds(raw,ids)
