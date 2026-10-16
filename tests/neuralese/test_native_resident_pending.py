"""Pending source debt uses the existing resident commit and CE progress frame."""
import json
import pytest
from test_native_resident import setup
from test_native_state_sft import graph
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.native_pending_sources import declaration
from natlang_neuralese.common.jsonio import write_json_durable as write


def pending_recipe(tmp_path, recipe, fingerprint='tokenizer-sha', *, only=False):
    row=graph('long');row.update(split='train',tokenizer_fingerprint=fingerprint)
    long=tmp_path/'long.jsonl';long.write_text(json.dumps(row)+'\n')
    pending=tmp_path/'pending.json';write(pending,declaration(long,long,
        tokenizer_fingerprint=fingerprint,role_ids=(),required_visits=2))
    value=json.loads(recipe.read_text())
    if only:
        value['input_bindings']['train']=value['input_bindings']['prior']
        value['input_bindings'].pop('prior')
        value['stages'][0]['inputs'].pop('continued_native_graphs')
    value['input_bindings']['pending']=resident.binding(pending)
    value['stages'][0]['inputs']['pending_native_sources']='pending';write(recipe,value)
    return pending


@pytest.mark.parametrize('only',[False,True])
@pytest.mark.parametrize('crash',[False,True])
def test_pending_adoption_commit_recovery_and_no_earned_credit(tmp_path,monkeypatch,only,crash):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    pending=pending_recipe(tmp_path,recipe,only=only)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    if crash:
        with monkeypatch.context() as patch:
            patch.setattr(resident,'repair',lambda *a: (_ for _ in ()).throw(OSError('projection crash')))
            with pytest.raises(RuntimeError,match='needs recovery'):manager.poll(7)
        committed=resident.recover(directory)
        from natlang_neuralese.data.native_sources import NativeGraphSource
        train=NativeGraphSource(paths['old'] if only else paths['new'],'tokenizer-sha',required_split='train')
        manager=resident.ResidentAdoption(directory,stage_id='sft',recipe_binding=committed['payload']['recipe'],
            train=train,held=manager.held)
    else:assert manager.poll(7) is not None
    loaded=manager.pending_declarations
    assert len(loaded)==1 and loaded[0]['binding']==resident.binding(pending)
    assert loaded[0]['rows']['long']['required_visits']==2
    assert 'exposure_counts' not in resident.current(directory)
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    from test_native_pending_sources import frame as training_frame
    training=training_frame()
    training.update({'options':{'stage_id':'sft','epochs':1,
        '_pending_native_source_declarations':loaded},'source_row_bindings':manager.train.bindings,
        'source_row_order':manager.train.ordered_ids(),'exposure_counts':{'a':1}})
    frame=collect_sft_progress(training)['sft']
    assert frame['exposure_counts']['a']==1 and frame['exposure_counts']['long']==0
    assert frame['required_visits']['long']==2


def test_mutated_pending_source_rejected_without_commit(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    _,directory,manager,recipe=setup(tmp_path)
    pending=pending_recipe(tmp_path,recipe,only=True)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    pending.write_text(pending.read_text()+'\n')
    assert manager.poll(2) is None and resident.current(directory) is None
    assert manager.train.ordered_ids()==['a','b']


def test_activated_pending_quota_is_scheduled_with_same_earned_counter():
    from natlang_neuralese.train.native_state_sft import pending_exposure_batches
    entries=[{'id':'fit','forward_tokens':3},{'id':'long','forward_tokens':7}]
    quotas,batches=pending_exposure_batches(entries,{'fit':1,'long':1},epochs=1,
        minimum_visits={'fit':1,'long':3,'still-pending':9})
    assert quotas=={'fit':1,'long':3}
    assert [(visit,indices) for visit,indices,_ in batches]==[(1,[1]),(2,[1])]


@pytest.mark.parametrize('target',['active','pending'])
def test_withdrawn_targets_rejected_before_commit_even_if_cpu_history_is_stale(tmp_path,monkeypatch,target):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    pending=pending_recipe(tmp_path,recipe,only=target=='pending')
    from natlang_neuralese.train.native_pending_sources import load_pending_sources
    from natlang_neuralese.data.native_sources import NativeGraphSource
    if target=='pending':
        row=load_pending_sources(pending,tokenizer_fingerprint='tokenizer-sha')['rows']['long']
        forbidden={'id':'long','binding':row['binding']}
    else:
        forbidden={'id':'c','binding':NativeGraphSource(paths['new'],'tokenizer-sha',required_split='train').bindings['c']}
    history=[{'proof_sha256':'f'*64,'rows':[forbidden]}]
    # These are the authoritative values supplied by the restored SFT owner.
    manager.withdrawal_history=history
    initial=json.loads((directory/'initial.json').read_text());initial['withdrawal_history']=history
    write(directory/'initial.json',initial)
    with pytest.raises(ValueError,match='previously withdrawn'):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert not (directory/'pending.json').exists()
    # A stale/malicious CPU request still cannot bypass the actual live owner.
    initial['withdrawal_history']=[];write(directory/'initial.json',initial)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert manager.poll(9) is None
    assert resident.current(directory) is None and manager.train.ordered_ids()==['a','b']
    rejection,=(directory/'rejections').glob('*.json')
    assert 'previously withdrawn' in json.loads(rejection.read_text())['error']
