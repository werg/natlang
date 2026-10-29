import json
import pytest
from natlang_neuralese.train.diloco import replica_baseline,replica_retention
from natlang_neuralese.train.diloco.replica_journal import _identity


def test_lease_and_interrupted_cleanup(tmp_path,monkeypatch):
    monkeypatch.setattr(replica_baseline,'verify',lambda *_:None)
    retention=replica_retention.Retention(tmp_path);ids=[str(n)*32 for n in range(1,5)]
    for i,g in enumerate(ids):
        path=tmp_path/('generation-'+g);(path/'journal').mkdir(parents=True);(path/'baseline').mkdir()
        identity={'run':i};(path/'journal'/'identity.json').write_text(json.dumps(identity));head=_identity(identity);base='a'*64
        pointer=tmp_path/'active.json';pointer.write_text(json.dumps({'id':g,'baseline_sha256':base,'head':head,'previous':{'sequence':2,'head':'f'*64}}))
        retention.register(generation=g,baseline_sha256=base,head=head,previous=ids[i-1] if i else None,replica_confirmation={'baseline_sha256':base,'new_head':head},active_pointer=pointer)
    retention.acquire(ids[0],'export-one');retention.collect()
    assert (tmp_path/('generation-'+ids[0])).exists() and not (tmp_path/('generation-'+ids[1])).exists()
    assert all((tmp_path/('generation-'+g)).exists() for g in ids[2:])
    retention.release('export-one',ids[0]);original=replica_retention.shutil.rmtree
    monkeypatch.setattr(replica_retention.shutil,'rmtree',lambda *_:(_ for _ in ()).throw(OSError('injected crash')))
    with pytest.raises(OSError):retention.collect()
    assert (tmp_path/('.retiring-'+ids[0])).is_dir()
    with pytest.raises(ValueError):retention.acquire(ids[0],'late-reader')
    monkeypatch.setattr(replica_retention.shutil,'rmtree',original)
    assert retention.collect()['remaining']==ids[2:]
    state=json.loads((tmp_path/'retention.json').read_text());assert len(state['retired'])==2
