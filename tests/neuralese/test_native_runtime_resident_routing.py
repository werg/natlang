"""A real drained source authority routes continuations without another decode."""
import json
from pathlib import Path
import pytest
from test_native_later_runtime_handoff import later
from natlang_neuralese.rl import native_runtime_handoff as runtime,native_raw_rounds as raw,native_raw_handoff as handoff,native_supervised as supervised
from natlang_neuralese.train import native_resident as resident


def test_actual_drained_directory_reaches_config_seed_and_next_continuation(later,monkeypatch):
 x=later;loaded=handoff.verify_handoff(x['handoff']);directory=loaded[0]['resident']['directory']
 assert directory!=str(Path(json.loads(Path(x['transition']['path']).read_bytes())['recipe']['path']).parent/'resident-data')
 monkeypatch.setattr(runtime,'verify_handoff',lambda *a,**k:pytest.fail('decoded full parent twice'))
 proof=runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],Path(x['proof']['path']),verified_handoff=loaded)
 assert proof==x['proof']
 path=raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'],verified_handoff=loaded)
 config=json.loads(path.read_bytes());assert config['resident_data_directory']==directory
 assert config['trainer_parameters']['resident_data_directory']==directory
 assert config['seed_trainer_parameters']['resident_data_directory']==directory
 recipe=json.loads(Path(config['production_recipe']).read_bytes());stage=next(s for s in recipe['stages']if s['id']==config['production_stage'])
 continuation,_=supervised.continuation_config(config,stage,checkpoint=loaded[0]['checkpoint']['path'],adapter=None,out=x['root']/'next-job')
 assert continuation['resident_data_directory']==directory
 # student_cycle routes its foundation trainer through the controller directory.
 assert raw.configure_model(x['plan'],x['declaration'],x['bootstrap'],x['root'],verified_handoff=loaded)==path
 assert runtime.verify_runtime_handoff(x['proof'],bootstrap=x['bootstrap'],verified_handoff=loaded)['checkpoint']==loaded[0]['checkpoint']


def test_loaded_pair_cannot_be_substituted_on_runtime_retry(later):
 x=later;record,saved=handoff.verify_handoff(x['handoff']);bad=dict(record);bad['checkpoint']=x['bootstrap']
 with pytest.raises(ValueError,match='reused handoff decode'):
  runtime.prepare_runtime_handoff(x['bootstrap'],x['handoff'],x['transition'],x['config'],Path(x['proof']['path']),verified_handoff=(bad,saved))
