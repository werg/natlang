import copy,json
from pathlib import Path
import pytest
from test_native_resident import setup
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.common import row_index

check_union=resident.check_index_witness_union

def witnesses(tmp_path,monkeypatch):
 monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'));monkeypatch.setenv('NATLANG_HASH_CACHE',str(tmp_path/'hashes.jsonl'))
 _,directory,manager,recipe=setup(tmp_path);request=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=());return json.loads(resident.bound(request).read_bytes())['payload']['indexes']

def test_29_requests_load_each_actual_index_once(tmp_path,monkeypatch):
 values=witnesses(tmp_path,monkeypatch);loaded=[];original=row_index.load
 def load(path):loaded.append(str(path));return original(path)
 monkeypatch.setattr(row_index,'load',load)
 assert check_union([copy.deepcopy(values)for _ in range(29)],resident._check_indexes)==len(values)
 assert len(loaded)==len(values)

@pytest.mark.parametrize('field',['metadata','key','cache'])
def test_conflicting_whole_witness_is_never_hidden(tmp_path,monkeypatch,field):
 values=witnesses(tmp_path,monkeypatch);bad=copy.deepcopy(values[0])
 if field=='metadata':bad['metadata_sha256']='0'*64
 elif field=='key':bad['key']['role_ids']=[123]
 else:bad['cache']['sha256']='0'*64
 with pytest.raises(ValueError):check_union([values,[bad],copy.deepcopy(values)],resident._check_indexes)

def test_changed_actual_cache_still_fails(tmp_path,monkeypatch):
 values=witnesses(tmp_path,monkeypatch);p=Path(values[0]['cache']['path']);p.write_bytes(p.read_bytes()+b'changed')
 with pytest.raises(ValueError):check_union([values,copy.deepcopy(values)],resident._check_indexes)
