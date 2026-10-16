"""Objective changes and append retain exact committed whole-episode credits."""
import copy
import json
import pytest
from natlang_neuralese.train.native_semantic_policy import PolicySource
from natlang_neuralese.common.hashing import sha256_file_hex


def declaration(tmp_path):
    def bind(name,value):
        path=tmp_path/name;path.write_text(value);return {'path':str(path),'sha256':sha256_file_hex(path)}
    rows=[{'task_id':'train','sample':sample,'group':'g','split':'train','turns':[{'actual':'bound in full compiler'}],
           'reward':None if reward is None else {'quality':reward}} for sample,reward in enumerate((1,0,None))]
    source={'rollouts':bind('rollouts.jsonl',''.join(json.dumps(row)+'\n' for row in rows)),
        'adapter':bind('adapter.json','{}'),'block_bundle':bind('bundle.json','{}')}
    return {'id':'policy','mode':'grpo','epochs':1,'every_semantic_updates':1,'updates_per_refresh':2,'sources':[source]},rows


def test_revision_does_not_reset_credit_or_inflate_unvisited_quota_and_resume_skips_completed(tmp_path):
    value,_=declaration(tmp_path);source=PolicySource(value);training={'step':0,'exposure':{}}
    source.synchronize(training);first=source.next(training)
    source.committed(training,*first);training['step']+=1
    saved=copy.deepcopy(training);tweaked={**value,'clip_high':.4};continued=PolicySource(tweaked,previous_sources=training['policy_source_history'])
    summary=continued.synchronize(training)
    counts=training['exposure']['policy'];quotas=training['policy_required_visits']['policy']
    assert counts[first[1]['id']]=={'episode':1} and quotas[first[1]['id']]==2
    untouched=next(entry for entry in source.entries if entry['id']!=first[1]['id'])
    assert quotas[untouched['id']]==1
    assert len(summary['omissions'])==1 and next(iter(summary['omissions'].values()))=='unscored'
    assert len(training['policy_objective_revisions'])==2
    for _ in range(2):continued.committed(training,*continued.next(training));training['step']+=1
    assert continued.next(training) is None and continued.summary(training)['complete']
    restored=PolicySource(tweaked,previous_sources=training['policy_source_history'])
    restored.synchronize(training)
    assert restored.next(training) is None
    assert training['exposure']['policy'][first[1]['id']]['episode']==2
    assert saved['exposure']['policy'][first[1]['id']]['episode']==1


def test_mode_change_cannot_erase_pending_failure_and_explicit_cancellation_never_earns_credit(tmp_path):
    value,_=declaration(tmp_path);source=PolicySource(value);training={'step':0,'exposure':{}}
    source.synchronize(training)
    changed={**value,'mode':'rft'}
    with pytest.raises(ValueError,match='explicit supervised cancellation'):
        PolicySource(changed).synchronize(training)
    changed['cancel_stale_reason']='supervisor switches to successful-task RFT and recollects failures'
    after=PolicySource(changed);summary=after.synchronize(training)
    assert len(summary['cancelled_uncredited'])==1 and not summary['original_obligations_complete']
    failed=next(iter(summary['cancelled_uncredited']))
    assert training['exposure']['policy'].get(failed,{})=={}
    after.committed(training,*after.next(training))
    assert after.summary(training)['complete'] and not after.summary(training)['original_obligations_complete']
    with pytest.raises(ValueError,match='recollect under current policy'):
        PolicySource(value).synchronize(training)


def test_pending_source_history_survives_append_without_data_loss_or_duplicate_admission(tmp_path):
    value,rows=declaration(tmp_path);first=PolicySource(value);training={'step':0,'exposure':{}}
    first.synchronize(training);first.committed(training,*first.next(training))
    extra=copy.deepcopy(rows[0]);extra.update(task_id='additional',sample=8)
    # RFT supports this genuinely successful singleton task; GRPO requires
    # actual scored group spread and retains flat groups as explicit omissions.
    other=tmp_path/'extra.jsonl';other.write_text(json.dumps(extra)+'\n')
    binding={**value['sources'][0],'rollouts':{'path':str(other),'sha256':sha256_file_hex(other)}}
    appended=PolicySource({**value,'sources':[binding]},previous_sources=training['policy_source_history'])
    appended.synchronize(training)
    assert len(appended.sources)==2 and len(appended.entries)==2
    assert len(appended.omissions)==2
    assert appended.summary(training)['remaining_episode_visits']==1
    source_path=tmp_path/'rollouts.jsonl';source_path.write_text(source_path.read_text()+'\n')
    with pytest.raises(ValueError,match='artifact changed'):
        PolicySource(value)
