"""Failed current readiness preserves earned state while useful SFT continues."""
import json
from pathlib import Path
import pytest
from natlang_neuralese.rl import native_raw_rounds as raw,native_rounds
from natlang_neuralese.common.jsonio import write_json_durable as write
from test_native_raw_rounds import fixture

@pytest.mark.parametrize('complete',[True,False])
def test_incomplete_readiness_publishes_checkpoint_and_retains_exact_failure(tmp_path,monkeypatch,complete):
 directory,config,stop,calls=fixture(tmp_path,monkeypatch)
 out=directory/'repair';out.mkdir();checkpoint=out/'checkpoint.pt';checkpoint.write_bytes(b'earned-optimizer-rng-exposure')
 report={'checkpoint':raw.binding(checkpoint),'foundation_ready':False,'completed':False,
  'communication_foundation':{'complete':complete},'task_success':0,'mandatory_sft':{'remaining':123}}
 write(out/'native-training-receipt.json',report)
 cursor={'foundation_qualified':True,'readiness':{'path':'old'}}
 raw._publish_supervised_result({'raw_foundation_qualification':True,'trainer_parameters':{}},out,config,cursor,{'pending_native_checkpoint':str(checkpoint)})
 assert raw._current_checkpoint(config)==raw.binding(checkpoint)
 assert 'readiness'not in cursor and 'foundation_qualified'not in cursor
 assert json.loads(Path(cursor['foundation_readiness_repair']['report']['path']).read_bytes())==report
 assert checkpoint.read_bytes()==b'earned-optimizer-rng-exposure'


def test_failed_foundation_retries_sft_without_policy_round_then_releases_only_real_success(tmp_path,monkeypatch):
 directory,config,stop,calls=fixture(tmp_path,monkeypatch)
 # Genuine current-owner import/reconciliation starts with SFT, not collection.
 write(directory/'cycle.json',{'cycle':0,'phase':'sft','next_stage':0})
 original=native_rounds.round_job;qualified=False
 def measured(*args):
  result=original(*args)
  if args[1].get('mandatory_sft_only'):
   path=Path(result['pending_native_checkpoint']).parent/'native-training-receipt.json'
   value=json.loads(path.read_bytes());value['foundation_ready']=qualified;value['completed']=False;value['task_success']=int(qualified);write(path,value)
  return result
 monkeypatch.setattr(native_rounds,'round_job',measured)
 first=raw.student_cycle('plan',{},None,directory,stop=stop)
 assert first['phase']=='sft'and first['cycle']==1 and 'foundation_qualified'not in first
 assert all(kind=='sft'for kind,_ in calls)
 failure=first['foundation_readiness_repair']['report'];failed_bytes=Path(failure['path']).read_bytes()
 second=raw.student_cycle('plan',{},None,directory,stop=stop)
 assert second['phase']=='sft'and second['cycle']==2 and all(kind=='sft'for kind,_ in calls)
 assert calls[-1][1]['trainer_parameters']['refresh_foundation_readiness']is True
 assert Path(failure['path']).read_bytes()==failed_bytes
 qualified=True
 third=raw.student_cycle('plan',{},None,directory,stop=stop)
 assert third['phase']=='rl'and third['foundation_qualified']and 'foundation_readiness_repair'not in third
 assert all(kind=='sft'for kind,_ in calls)
 raw.student_cycle('plan',{},None,directory,stop=stop)
 assert any(kind=='rl'for kind,_ in calls)


def test_fresh_post_stage_failure_revokes_prior_foundation_release(tmp_path,monkeypatch):
 directory,config,stop,calls=fixture(tmp_path,monkeypatch)
 write(directory/'cycle.json',{'cycle':4,'phase':'sft','next_stage':0,'foundation_qualified':True})
 original=native_rounds.round_job
 def failed_current(*args):
  result=original(*args)
  if args[1].get('raw_foundation_qualification'):
   path=Path(result['pending_native_checkpoint']).parent/'native-training-receipt.json'
   value=json.loads(path.read_bytes());value.update(foundation_ready=False,completed=False);write(path,value)
  return result
 monkeypatch.setattr(native_rounds,'round_job',failed_current)
 cursor=raw.student_cycle('plan',{},None,directory,stop=stop)
 assert cursor['phase']=='sft'and cursor['cycle']==5
 assert 'foundation_qualified'not in cursor and 'readiness'not in cursor
 assert cursor['foundation_readiness_repair']['checkpoint']==raw._current_checkpoint(config)
 assert all(kind=='sft'for kind,_ in calls)
