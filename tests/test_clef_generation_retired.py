import json

import pytest

from scripts.neuralese_data import clef_pools_daily as daily


def test_retired_generation_stops_before_provider_or_output_mutation(tmp_path,monkeypatch):
    monkeypatch.setattr(daily,'CORPORA',tmp_path)
    def forbidden(*args,**kwargs):
        raise AssertionError('retired generation attempted a subprocess/provider/registration')
    monkeypatch.setattr(daily.subprocess,'run',forbidden)
    monkeypatch.setattr(daily,'registered_ids',forbidden)
    monkeypatch.setattr(daily,'register_entries',forbidden)
    with pytest.raises(RuntimeError,match='External teacher generation is retired'):
        daily.run('20261017',800000,700)
    assert list(tmp_path.iterdir())==[]


def test_existing_source_registration_remains_available_without_generating_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(daily,'CORPORA',tmp_path)
    monkeypatch.setattr(daily,'ROOT',tmp_path)
    day='20261016';directory=daily.batch_dir(day);directory.mkdir()
    entry={'id':daily.corpus_id(day),'source_class':'historical external labels'}
    (directory/'registry-entry.json').write_text(json.dumps(entry))
    monkeypatch.setattr(daily,'registered_ids',lambda:set())
    registrations=[]
    monkeypatch.setattr(daily,'register_entries',lambda entries,message:registrations.extend(entries))
    daily.register(day)
    assert registrations==[entry]
