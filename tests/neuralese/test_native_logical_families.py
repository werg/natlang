import hashlib,json
from types import SimpleNamespace
import pytest
from natlang_neuralese.train.native_logical_families import load_map
from natlang_neuralese.train.own_output_supervision import family_scales

def test_logical_map_binds_each_row_and_source(tmp_path):
    entries=[{'id':'a','binding':'a'*64},{'id':'b','binding':'b'*64}]
    source=SimpleNamespace(file_sha256='c'*64,entries=entries)
    path=tmp_path/'families.json'
    doc={'schema':'natlang.logical-family-map/1','source_sha256':source.file_sha256,
         'rows':[{'id':'a','binding':'a'*64,'family':'communication'},
                 {'id':'b','binding':'b'*64,'family':'coding'}]}
    path.write_text(json.dumps(doc));ref={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert load_map(ref,source)==['communication','coding']
    with pytest.raises(ValueError,match='source differs'):
        load_map(ref,SimpleNamespace(file_sha256='d'*64,entries=entries))
    changed=SimpleNamespace(file_sha256=source.file_sha256,entries=list(reversed(entries)))
    with pytest.raises(ValueError,match='row identity'):
        load_map(ref,changed)

def test_bounded_mean_one_scales_ignore_chunk_boundaries():
    entries=[{'id':str(i)} for i in range(101)]
    ids=['communication']*100+['coding']
    objective={'family_balance':{'key':'logical-family','max_scale':8.,'normalization':'bounded-mean-one/1','map':{}}}
    first,census=family_scales(objective,entries,logical_ids=ids,cohort_ids=['file-a']*101)
    second,_=family_scales(objective,entries,logical_ids=ids,cohort_ids=['file-'+str(i) for i in range(101)])
    assert first==second and max(first.values())<=8.
    assert sum(first.values())==pytest.approx(len(entries))
    assert census['families']['coding']['scale']==8.
