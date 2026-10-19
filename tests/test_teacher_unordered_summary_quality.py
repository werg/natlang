import copy
import json
from pathlib import Path
import pytest
from scripts.neuralese_data import teacher_file_quality as q


def fixture(tmp_path, change=None):
    expected={'ids':['a','b'],'total':9,'count':2}
    row={'id':'captured:one','task':{'program_ir':{'id':'program:one','split':'train',
        'source_groups':['fresh-group'],'family':'curriculum_short_execution_recovery',
        'semantics':{'expected':expected,'expected_files':{'summary.json':json.dumps(expected)},
                     'folder_files':{'summary.json':'{}'}}}},
         'outcome':{'status':'done','accepted':False,'oracle':{'accepted':False,'level':'exact'},
                    'files_check':{'accepted':False,'failed':['summary.json']},
                    'quality_pending':[],'rejection_reasons':['files','answer'],
                    'checks':{key:key not in ('answer','files')for key in ('seeded_failure_requirement',
                        'expected_status','honest_stop','effects','files','world','file_return_consistency','answer')},
                    'value':{'ids':['b','a'],'total':9,'count':2},
                    'files':{'summary.json':json.dumps({'ids':['b','a'],'total':9,'count':2})}},
         'trajectory':[{'model_response':'original own teacher'}]}
    if change:change(row)
    def put(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value)+'\n')
        return {'path':str(path),'sha256':q.sha256_file_hex(path),'bytes':path.stat().st_size}
    original=put('original.jsonl',row)
    identity={'id':row['id'],'original_row_sha256':q.canonical_json_sha256_hex(row),
              'program_sha256':q.canonical_json_sha256_hex(row['task']['program_ir'])}
    inventory=put('inventory.json',{'schema':'natlang.teacher-order-contract-review-inventory/1',
        'training_admission':False,'rows':[{**identity,'original':original,
                       'expected':row['task']['program_ir']['semantics']['expected']}]})
    review=put('review.json',{'schema':'natlang.teacher-unordered-summary-semantic-review/1',
        'reviewer':'independent-root-source-meaning-review','training_admission':False,
        'reason_verbatim':'The source requires membership and values, not ID order.',
        'inventory':inventory,'approved':[identity]})
    return row,original,review


def test_order_reverification_preserves_original_and_replays(tmp_path):
    row,original,review=fixture(tmp_path);before=copy.deepcopy(row)
    derived=q.continue_unordered_summary_quality(row,original,review_reference=review)
    assert row==before and derived['task']==row['task'] and derived['trajectory']==row['trajectory']
    assert derived['outcome']['accepted'] and not derived['outcome']['files_check']['failed']
    q.verify_quality_continuation(derived,row,original)
    derived['trajectory'][0]['model_response']='rewritten'
    with pytest.raises(ValueError,match='differs'):q.verify_quality_continuation(derived,row,original)


@pytest.mark.parametrize('bad',['duplicate','wrong_total','bool_count','extra_key','file_mismatch',
    'extra_file','held','other_check','missing_check','pending','wrong_family'])
def test_even_approved_row_cannot_waive_hard_contract(tmp_path,bad):
    def change(row):
        o=row['outcome'];p=row['task']['program_ir']
        if bad=='duplicate':o['value']['ids']=['a','a']
        elif bad=='wrong_total':o['value']['total']=10
        elif bad=='bool_count':o['value']['count']=True
        elif bad=='extra_key':o['value']['extra']=0
        elif bad=='file_mismatch':o['files']['summary.json']=json.dumps(p['semantics']['expected'])
        elif bad=='extra_file':o['files']['extra']='x'
        elif bad=='held':p['split']='test'
        elif bad=='other_check':o['checks']['effects']=False
        elif bad=='missing_check':del o['checks']['world']
        elif bad=='pending':o['quality_pending']=['review']
        elif bad=='wrong_family':p['family']='unrelated'
    row,original,review=fixture(tmp_path,change)
    with pytest.raises(ValueError):q.continue_unordered_summary_quality(row,original,review_reference=review)


def test_tampered_authority_or_source_is_rejected(tmp_path):
    row,original,review=fixture(tmp_path)
    Path(review['path']).write_text('{}')
    with pytest.raises(ValueError):q.continue_unordered_summary_quality(row,original,review_reference=review)


def test_actual_converter_enriched_original_binding_is_reconstructed(tmp_path):
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import verify_original_row
    row,physical,review=fixture(tmp_path)
    enriched={**physical,'source':'/work/original.jsonl','row_id':row['id'],'positive':True}
    derived=q.continue_unordered_summary_quality(row,enriched,review_reference=review)
    assert verify_original_row(derived,enriched)==Path(physical['path'])
    with pytest.raises(ValueError):verify_original_row(derived,{**enriched,'source':'different'})
