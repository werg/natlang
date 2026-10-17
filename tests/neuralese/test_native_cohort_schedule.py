"""Newly admitted sources receive SGD without losing old complete-recipe debt."""
from collections import Counter
import copy
import json

import pytest

from natlang_neuralese.train.native_cohort_schedule import source_cohort_ids
from natlang_neuralese.train.native_state_sft import pending_exposure_batches, build_parser
from natlang_neuralese.train.native_source_streams import stream_pending_batches
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_jsonl_text
from test_native_state_sft import graph
from test_native_source_streams import count_readers


def rows():
    return ([dict(id=f'old-{i}', forward_tokens=10, training_priority=3) for i in range(100)]
            + [dict(id=f'new-{i}', forward_tokens=10, training_priority=3) for i in range(8)])


def names(entries, batches):
    return [entries[i]['id'] for _, indices, _ in batches for i in indices]


def test_new_cohort_no_longer_waits_behind_old_source_and_resume_keeps_every_credit():
    entries=rows();cohorts=['old']*100+['new']*8;credits={f'old-{i}':1 for i in range(10)}
    before=copy.deepcopy(entries)
    _,old=pending_exposure_batches(entries,credits,epochs=1)
    assert names(entries,old)[0]=='old-10'
    quotas,batches=pending_exposure_batches(entries,credits,epochs=1,cohort_ids=cohorts)
    assert names(entries,batches)[:8]==[f'new-{i}' for i in range(8)]
    for _,indices,_ in batches[:3]:
        for i in indices:credits[entries[i]['id']]=credits.get(entries[i]['id'],0)+1
    entries.append(dict(id='fresh',forward_tokens=99,training_priority=4));cohorts.append('fresh')
    quotas,remaining=pending_exposure_batches(entries,credits,epochs=1,batch_tokens=25,
        cohort_ids=cohorts,schedule_offset=15)
    assert names(entries,remaining)[0]=='fresh'
    assert Counter(names(entries,remaining))=={e['id']:quotas[e['id']]-credits.get(e['id'],0)
        for e in entries if quotas[e['id']]>credits.get(e['id'],0)}
    assert any(indices==[108] for _,indices,_ in remaining)
    assert entries[:108]==before


def test_token_service_balances_work_and_priority_breaks_ties():
    entries=[dict(id='long',forward_tokens=100,training_priority=4)]+[
        dict(id=f'short-{i}',forward_tokens=10,training_priority=3) for i in range(20)]
    _,batches=pending_exposure_batches(entries,{},epochs=2,cohort_ids=['long']+['short']*20)
    actual=names(entries,batches)
    assert actual[0]=='long'
    assert actual[1:11]==[f'short-{i}' for i in range(10)]
    assert actual[11]=='long'
    for index,(visit,_,last) in enumerate(batches):
        assert last==all(other!=visit for other,_,_ in batches[index+1:])


def test_foundation_and_zero_priority_debt_lanes_keep_their_service():
    entries=([dict(id=f'comm-{i}',forward_tokens=10,communication=True)for i in range(20)]
        +rows()+[dict(id=f'debt-{i}',forward_tokens=10)for i in range(10)])
    cohorts=['comm']*20+['old']*100+['new']*8+['debt']*10
    foundation={e['id'] for e in entries[:20]}
    quotas,batches=pending_exposure_batches(entries,{},epochs=1,cohort_ids=cohorts,
        foundation_ids=foundation,foundation_communication_batches=4)
    actual=names(entries,batches)
    assert actual[:5]==['comm-0','comm-1','comm-2','comm-3','debt-0']
    assert 'new-0' in actual[:12]
    assert Counter(actual)==quotas


def real_source(tmp_path):
    bindings=[]
    for cohort in ('old','new'):
        path=tmp_path/(cohort+'.jsonl.zst')
        with write_jsonl_text(path)as stream:
            for i in range(8):
                stream.write(json.dumps(dict(graph(f'{cohort}-{i}'),tokenizer_fingerprint='1'*64))+'\n')
        bindings.append(dict(kind='native-graphs',path=path.name,sha256=sha256_file_hex(path),training_priority=3))
    path=tmp_path/'sources.json';path.write_text(json.dumps(dict(schema='natlang.native-graph-sources/1',sources=bindings)))
    return NativeGraphSource(path,'1'*64,required_split='train',role_ids=(58,59,60,61))


def test_real_equal_rank_compressed_cohorts_resume_without_rescans_or_lost_rows(tmp_path,monkeypatch):
    source=real_source(tmp_path);cohorts=source_cohort_ids(source)
    entries=copy.deepcopy(source.entries);credits={'old-0':1,'old-1':1}
    quotas,batches=pending_exposure_batches(source.entries,credits,epochs=2,batch_graphs=2,cohort_ids=cohorts)
    assert names(source.entries,batches)[0]=='new-0'
    counts,active=count_readers(monkeypatch)
    actual=list(stream_pending_batches(source,batches,cohort_ids=cohorts))
    assert [e.id for _,_,_,batch in actual for e in batch]==names(source.entries,batches)
    assert Counter(e.id for _,_,_,batch in actual for e in batch)=={
        e['id']:quotas[e['id']]-credits.get(e['id'],0)for e in source.entries}
    assert set(counts.values())=={2} and not active
    assert source.entries==entries


def test_nested_cohort_identity_survives_wrapping_and_rejects_mutation(tmp_path):
    source=real_source(tmp_path);before=source_cohort_ids(source)
    outer=tmp_path/'outer.json';outer.write_text(json.dumps(dict(schema='natlang.native-graph-sources/1',sources=[
        dict(kind='native-graphs',path=source.path.name,sha256=source.file_sha256)])))
    wrapped=NativeGraphSource(outer,'1'*64,required_split='train',role_ids=(58,59,60,61))
    assert source_cohort_ids(wrapped)==before
    source.path.write_text(source.path.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):source_cohort_ids(wrapped)


def test_fairness_is_explicit_recipe_parameter_and_saved_trainer_option():
    from natlang_neuralese.train.recipe import HANDLERS
    assert 'urgent_cohort_fairness' in HANDLERS['native_state_sft']['parameters']
    args=build_parser().parse_args(['--native-graphs','a','--held-native-graphs','b','--out','c','--urgent-cohort-fairness'])
    assert vars(args)['urgent_cohort_fairness']is True


def test_existing_unfinished_recipe_accepts_explicit_fairness_tuning(tmp_path):
    from natlang_neuralese.train.recipe import direct_windowing_handoffs
    before={'stages':[{'id':'sft','kind':'native_state_sft','parameters':{'epochs':1}}]}
    after=copy.deepcopy(before)
    after['stages'][0]['parameters']['urgent_cohort_fairness']=True
    changes,old_identity,new_identity=direct_windowing_handoffs(before,after,tmp_path)
    assert changes=={'sft':{'urgent_cohort_fairness':[None,True]}}
    assert old_identity==new_identity


@pytest.mark.parametrize('outer_priority', [3, 4])
def test_positive_nested_bundle_services_teacher_leaf_and_resumes_exact_readers(tmp_path, monkeypatch, outer_priority):
    source = real_source(tmp_path)
    leaf_ids = source_cohort_ids(source)
    outer = tmp_path / 'positive-bundle.json'
    outer.write_text(json.dumps(dict(schema='natlang.native-graph-sources/1', sources=[
        dict(kind='native-graphs', path=source.path.name, sha256=source.file_sha256,
             training_priority=outer_priority)])))
    wrapped = NativeGraphSource(outer, '1'*64, required_split='train', role_ids=(58,59,60,61))
    cohorts = source_cohort_ids(wrapped)
    assert cohorts == leaf_ids
    assert len(set(cohorts)) == 2
    entries = copy.deepcopy(wrapped.entries)
    credits = {'old-0': 1, 'old-1': 1}
    saved = copy.deepcopy(credits)
    quotas, batches = pending_exposure_batches(wrapped.entries, credits, epochs=2,
        batch_graphs=2, cohort_ids=cohorts)
    assert names(wrapped.entries, batches)[0] == 'new-0'
    counts, active = count_readers(monkeypatch)
    actual = list(stream_pending_batches(wrapped, batches, cohort_ids=cohorts))
    assert [e.id for _,_,_,batch in actual for e in batch] == names(wrapped.entries, batches)
    assert set(counts.values()) == {2} and not active
    assert Counter(names(wrapped.entries, batches)) == {
        e['id']: quotas[e['id']] - saved.get(e['id'], 0) for e in entries}
    assert credits == saved and wrapped.entries == entries
    # A real saved prefix of this schedule resumes from those exact row credits.
    for _, indices, _ in batches[:3]:
        for index in indices:
            row_id = entries[index]['id']
            credits[row_id] = credits.get(row_id, 0) + 1
    _, remaining = pending_exposure_batches(wrapped.entries, credits, epochs=2,
        batch_graphs=2, cohort_ids=source_cohort_ids(wrapped))
    assert Counter(names(entries, remaining)) == {
        e['id']: quotas[e['id']] - credits.get(e['id'], 0)
        for e in entries if quotas[e['id']] > credits.get(e['id'], 0)}
    # Positive wrappers must still authenticate the interior manifest itself.
    source.path.write_text(source.path.read_text() + ' ')
    with pytest.raises(ValueError, match='changed'):
        source_cohort_ids(wrapped)
