"""Whole preferences retain exact objectives and real aligned KV allocation."""
import json
from pathlib import Path
import pytest

from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.rl.native_replay import graph_identity
from natlang_neuralese.train.native_preference import SCHEMA, KIND, NativePreferenceObjective
from scripts.neuralese_data.partition_acquired_native_capacity import partition_preferences, main


def ref(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file_hex(path)}


def fixture(tmp_path):
    root=tmp_path/'source';root.mkdir()
    models={};entrypoints={};original={}
    for model in ('lfm','mellum'):
        rows={};proofs={};outputs=[]
        for split,widths in (('train',(2,129)),('test',(3,))):
            path=root/(model+'-'+split+'.jsonl');lines=[]
            for width in widths:
                source_sha=digest([model,split,width]);nodes=[];suffixes={}
                for name in ('chosen','rejected'):
                    ids=list(range(width if name=='chosen' else 2))
                    nodes.append({'name':name+':0','token_ids':ids,'target_weights':[0.]*len(ids)})
                    suffixes[name+':0']={'alternative':name,'suffix_start':1,'suffix_stop':len(ids),'token_ids_sha256':digest(ids)}
                counts={name:s['suffix_stop']-s['suffix_start'] for name,s in ((n.split(':')[0],s)for n,s in suffixes.items())}
                identifier='offline-preference:'+digest({'source_record_sha256':source_sha,'suffixes':suffixes,'token_counts':counts})
                row={'schema':'natlang.neuralese-native-graph/1','id':identifier,'split':split,
                     'source_groups':[model+'-'+split], 'tokenizer_fingerprint':'fp','nodes':nodes,
                     'required_objective':KIND,'conversion':{'source_record_sha256':source_sha}}
                example=parse_graph(row,tokenizer_fingerprint='fp')
                rows[identifier]={'graph_sha256':digest(graph_identity(example)),'source_record_sha256':source_sha,
                                  'split':split,'suffixes':suffixes,'token_counts':counts}
                proofs[identifier]={'graph_sha256':digest(row),'source_record_sha256':source_sha,
                                    'split':split,'source_groups':row['source_groups']}
                line=json.dumps(row,separators=(', ',': '))+'\n'
                lines.append(line);original[model,identifier]=line
            path.write_text(''.join(lines));outputs.append({**ref(path),'rows':len(widths),'split':split})
        objective=root/(model+'-objective.json')
        objective.write_text(json.dumps({'schema':SCHEMA,'kind':KIND,'beta':.1,'margin':0.,'offline':True,'native_on_policy':False,'rows':rows}))
        held=root/(model+'-held-sources.json');held.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[
            {'kind':'native-graphs','path':Path(output['path']).name,'sha256':output['sha256'],'weight':1.,'epochs':1}
            for output in outputs if output['split']=='test']}))
        models[model]=[{'objective':ref(objective),'outputs':outputs,'proofs':proofs,'source_render_verified':True}]
        entrypoints[model]={'objective_spec':ref(objective),'held_native_graphs':ref(held)}
    audit=root/'audit.json';audit.write_text(json.dumps({'schema':'natlang.acquired-offline-preference-admission/1',
        'status':'passed','all_pair_token_render_verified':True,'models':models,'training_entrypoints':entrypoints}))
    return audit,original


def test_actual_padding_full_pair_conservation_and_usable_objectives(tmp_path):
    audit,original=fixture(tmp_path);out=tmp_path/'out'
    limits={m:{'forward_tokens':200,'kv_slots':150}for m in ('lfm','mellum')}
    receipt=partition_preferences(audit,limits,out)
    assert all(c['train:fit']['rows']==c['train:remaining-long']['rows']==1 for c in receipt['counts'].values())
    assert receipt['execution_qualified']is False and receipt['earned_pair_visits']==0
    seen={}
    for file in (out/'files').glob('*/*/*.jsonl.zst'):
        with open_jsonl_text(file)as stream:
            for line in stream:
                row=json.loads(line);key=file.parent.parent.name,row['id']
                assert key not in seen and line==original[key]
                seen[key]=line
    assert seen==original
    coverage=[json.loads(s)for s in(out/receipt['coverage']).read_text().splitlines()]
    assert all(r['geometry']['kv_slots']>r['geometry']['forward_tokens']for r in coverage)
    pending=json.loads((out/receipt['pending_source_declaration']).read_text())
    assert len(pending['rows'])==2 and all(r['required_visits']==1 and 'completed_visits'not in r for r in pending['rows'])
    for model in limits:
        for part in ('fit','remaining-long'):
            objective=NativePreferenceObjective(out/receipt['objectives'][model][part])
            assert len(objective.rows)==2 # selected train pair plus every held pair
            manifest=json.loads((out/receipt['manifests'][model]['train:'+part]).read_text())
            source=(out/receipt['manifests'][model]['train:'+part]).parent/manifest['sources'][0]['path']
            with open_jsonl_text(source)as stream:
                for line in stream:
                    row=json.loads(line)
                    objective.forward_inputs({'examples':[parse_graph(row,tokenizer_fingerprint='fp')]})
    assert partition_preferences(audit,limits,out)==receipt


def test_token_sum_cannot_admit_pair_beyond_real_kv_boundary(tmp_path):
    audit,_=fixture(tmp_path)
    receipt=partition_preferences(audit,{m:{'forward_tokens':4,'kv_slots':4}for m in('lfm','mellum')},tmp_path/'out')
    assert all(c['train:fit']['rows']==0 and c['train:remaining-long']['rows']==2 for c in receipt['counts'].values())


@pytest.mark.parametrize('change',['source','output','objective','held'])
def test_resume_checks_all_immutable_dependencies(tmp_path,change):
    audit,_=fixture(tmp_path);out=tmp_path/'out';limits={m:{'forward_tokens':200,'kv_slots':150}for m in('lfm','mellum')}
    receipt=partition_preferences(audit,limits,out);data=json.loads(audit.read_text())
    path={'source':Path(data['models']['lfm'][0]['outputs'][0]['path']),
          'objective':Path(data['models']['lfm'][0]['objective']['path']),
          'held':Path(data['training_entrypoints']['lfm']['held_native_graphs']['path']),
          'output':out/receipt['files'][0]['path']}[change]
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='immutable'):partition_preferences(audit,limits,out)


@pytest.mark.parametrize('change',['proof','missing','objective-row','split'])
def test_inconsistent_source_join_fails_before_publication(tmp_path,change):
    audit,_=fixture(tmp_path);data=json.loads(audit.read_text());view=data['models']['lfm'][0]
    if change=='proof':next(iter(view['proofs'].values()))['graph_sha256']='0'*64
    elif change=='missing':view['proofs'].pop(next(iter(view['proofs'])))
    elif change=='split':view['outputs'][0]['split']='test'
    else:
        path=Path(view['objective']['path']);obj=json.loads(path.read_text());next(iter(obj['rows'].values()))['token_counts']['chosen']+=1
        path.write_text(json.dumps(obj));view['objective']=ref(path);data['training_entrypoints']['lfm']['objective_spec']=ref(path)
    audit.write_text(json.dumps(data));out=tmp_path/'out'
    with pytest.raises(ValueError):partition_preferences(audit,{m:{'forward_tokens':200,'kv_slots':150}for m in('lfm','mellum')},out)
    assert not(out/'receipt.json').exists()


def test_cli_exposes_complete_preference_partition_without_session_samples(tmp_path):
    audit,_=fixture(tmp_path);limits=tmp_path/'limits.json';limits.write_text(json.dumps({m:{'forward_tokens':200,'kv_slots':150}for m in('lfm','mellum')}))
    main(['--audit',str(audit),'--preferences','--limits',str(limits),'--out',str(tmp_path/'out')])
    assert(tmp_path/'out/receipt.json').exists()


@pytest.mark.parametrize('change',['missing','epochs','weight','duplicate','boolean-epochs','boolean-weight'])
def test_held_manifest_must_expose_exact_held_pair_sources(tmp_path,change):
    audit,_=fixture(tmp_path);data=json.loads(audit.read_text())
    path=Path(data['training_entrypoints']['lfm']['held_native_graphs']['path'])
    manifest=json.loads(path.read_text())
    if change=='missing':manifest['sources']=[]
    elif change=='duplicate':manifest['sources']*=2
    elif change.startswith('boolean-'):manifest['sources'][0][change.removeprefix('boolean-')]=True
    else:manifest['sources'][0][change]=2
    path.write_text(json.dumps(manifest));data['training_entrypoints']['lfm']['held_native_graphs']=ref(path)
    audit.write_text(json.dumps(data));out=tmp_path/'out'
    with pytest.raises(ValueError,match='held'):
        partition_preferences(audit,{m:{'forward_tokens':200,'kv_slots':150}for m in('lfm','mellum')},out)
    assert not(out/'receipt.json').exists()
