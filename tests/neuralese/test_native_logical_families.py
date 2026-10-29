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

def test_append_keeps_old_labels_and_rejects_tampering(tmp_path):
    path=tmp_path/'families.json'
    def make(source, rows):
        path.write_text(json.dumps({'schema':'natlang.logical-family-map/1',
            'source_sha256':source.file_sha256,'rows':rows}))
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    a={'id':'a','binding':'a'*64,'family':'structural-iteration'}
    b={'id':'b','binding':'b'*64,'family':'native-communication'}
    c={'id':'c','binding':'c'*64,'family':'unclassified'}
    src=lambda digest,rows:SimpleNamespace(file_sha256=digest,entries=[
        {'id':r['id'],'binding':r['binding']} for r in rows])
    old=src('c'*64,[a,b]);new=src('d'*64,[a,b,c])
    assert load_map(make(old,[a,b]),old)==load_map(make(new,[a,b,c]),new)[:2]
    ref=make(new,[a,b,c]);path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='SHA mismatch'):load_map(ref,new)

def test_token_mass_normalization_equalizes_supervised_tokens_not_rows():
    # One long coding trajectory against many short communication rows: under token mass the coding row already
    # carries half the tokens, so it must not also receive the row-count cap.
    entries=[{'id':f'c{i}','supervised_tokens':10} for i in range(100)]+[{'id':'code','supervised_tokens':1000}]
    ids=['communication']*100+['coding']
    balance={'key':'logical-family','max_scale':8.,'normalization':'token-mass-mean-one/1','map':{}}
    scales,census=family_scales({'mass':'token','family_balance':balance},entries,logical_ids=ids)
    assert scales['code']==pytest.approx(1.) and scales['c0']==pytest.approx(1.)
    assert census['supervised_tokens']==2000 and census['families']['coding']['supervised_tokens']==1000
    entries[-1]['supervised_tokens']=4000  # coding now dominates: scaled down, token mass still mean one
    scales,_=family_scales({'mass':'token','family_balance':balance},entries,logical_ids=ids)
    assert scales['code']*4000==pytest.approx(scales['c0']*1000)
    assert sum(scales[e['id']]*e['supervised_tokens'] for e in entries)==pytest.approx(5000)
    with pytest.raises(ValueError,match="mass: 'token'"):
        family_scales({'mass':'invocation','family_balance':balance},entries,logical_ids=ids)
