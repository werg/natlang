import copy
import json
from pathlib import Path
import pickle

import pytest

from scripts.neuralese_data import continue_native_source_indexes as c
from natlang_neuralese.train import native_source_classification as m, recipe
from test_source_objective_index_rekey import case
from test_native_supervision_mass import declared


def test_index_certificate_reuses_digest_but_loads_and_checks_authority(combined, monkeypatch):
    from natlang_neuralese.train import native_classification_continuation as continuation
    _,_,_,prior,*_=combined
    part=prior['indexes'][0]; continuation._INDEX_CERTIFICATES.clear()
    digests=[];loads=[];original_digest=continuation.digest;original_load=m.row_index.load
    def digest(value):
        digests.append(1);return original_digest(value)
    def load(path):
        loads.append(path);return original_load(path)
    monkeypatch.setattr(continuation,'digest',digest);monkeypatch.setattr(m.row_index,'load',load)
    first=continuation._indexed(part,'new_index')
    assert continuation._indexed(part,'new_index')==first
    assert len(digests)==1 and len(loads)==2
    with pytest.raises(ValueError):continuation._indexed({**part,'rows':part['rows']+1},'new_index')
    for field,value in [('sha256','0'*64),('bytes',part['new_index']['bytes']+1)]:
        with pytest.raises(ValueError):
            continuation._indexed({**part,'new_index':{**part['new_index'],field:value}},'new_index')


@pytest.mark.parametrize('fault',['same-mtime','replacement','metadata','during-load'])
def test_index_certificate_rejects_mutation_or_recertifies_replacement(combined,monkeypatch,fault):
    import os
    from natlang_neuralese.train import native_classification_continuation as continuation
    _,_,_,prior,*_=combined;part=copy.deepcopy(prior['indexes'][0]);path=Path(part['new_index']['path'])
    continuation._INDEX_CERTIFICATES.clear();continuation._indexed(part,'new_index')
    raw=path.read_bytes();state=path.stat()
    if fault=='same-mtime':
        path.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(path,ns=(state.st_atime_ns,state.st_mtime_ns))
    elif fault=='metadata':
        value=pickle.loads(raw);value['metadata_sha256']='0'*64
        path.write_bytes(pickle.dumps(value));part['new_index']=m._binding(path)
    elif fault=='replacement':
        replacement=path.with_suffix('.replacement');replacement.write_bytes(raw)
        os.utime(replacement,ns=(state.st_atime_ns,state.st_mtime_ns));replacement.replace(path)
        digests=[];original=continuation.digest
        def digest(value):digests.append(1);return original(value)
        monkeypatch.setattr(continuation,'digest',digest)
        continuation._indexed(part,'new_index');assert len(digests)==1
        return
    else:
        original=m.row_index.load
        def load(target):
            value=original(target);target.write_bytes(raw);return value
        monkeypatch.setattr(m.row_index,'load',load)
    with pytest.raises(ValueError):continuation._indexed(part,'new_index')


def test_index_certificate_cache_is_bounded(combined,monkeypatch):
    from natlang_neuralese.train import native_classification_continuation as continuation
    _,_,_,prior,*_=combined;continuation._INDEX_CERTIFICATES.clear()
    monkeypatch.setattr(continuation,'_INDEX_CERTIFICATE_LIMIT',1)
    continuation._indexed(prior['indexes'][0],'new_index')
    continuation._indexed(prior['held_indexes'][0],'new_index')
    assert len(continuation._INDEX_CERTIFICATES)==1


@pytest.fixture
def combined(case, tmp_path):
    oldpkg,newpkg,source,bind,entries,value,write,index=case
    def put(path,v):
        path.write_text(json.dumps(v));return bind(path)
    def index_ref(path,v):
        with path.open('wb')as f:pickle.dump(v,f)
        return bind(path)
    oldref=write()
    oldpart={'source':bind(source),'old_index':oldref,'new_index':oldref,'rows':len(entries),
        'producer':bind(oldpkg/'train/native_source_classification.py'),
        'old_policy':{n:bind(oldpkg/n)for n in c.POLICY_FILES},'ordinary_partition':None}
    held=tmp_path/'held.jsonl';held.write_text('separate protected source\n')
    he=[{**entries[0],'id':'held','binding':'d'*64}];hv={'schema':'natlang.row-index/1',
        'key':{**value['key'],'source_sha256':bind(held)['sha256'],'split':'test'},'entries':he,'metadata_sha256':c.digest(he)}
    hi=index_ref(tmp_path/'held.pkl',hv);hp={**oldpart,'source':bind(held),'old_index':hi,'new_index':hi,'rows':1}
    oldtrain=put(tmp_path/'old-train.json',{'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs',**bind(source)}]})
    filtered=tmp_path/'filtered.jsonl';filtered.write_text('exact retained raw source fixture\n');retained=entries[1:]
    fv={**value,'key':{**value['key'],'source_sha256':bind(filtered)['sha256']},'entries':retained,'metadata_sha256':c.digest(retained)}
    fi=index_ref(tmp_path/'filtered.pkl',fv)
    removed={'id':entries[0]['id'],'binding':entries[0]['binding'],'source_sha256':bind(source)['sha256'],'reason':'exact unproductive ordinary decision'}
    filt=put(tmp_path/'filter.json',{'schema':'natlang.native-source-exact-withdrawal-filter/1','source_sha256':bind(source)['sha256'],
        'output':bind(filtered),'old_rows':3,'new_rows':2,'old_ids_sha256':c.digest([e['id']for e in entries]),
        'new_ids_sha256':c.digest([e['id']for e in retained]),'removed':[{**removed,'raw_row_sha256':'e'*64}]})
    ev=put(tmp_path/'evidence.json',{'decision':'supervisor exact rejected ordinary target'})
    wd=put(tmp_path/'withdrawal.json',{'schema':'natlang.native-graph-withdrawal/1','model_id':'fixture:model','evidence':ev,'rows':[removed]})
    fp=c.continue_classified_objective_index(bind(filtered),fi,old_package=oldpkg,new_package=newpkg)
    heldpart=c.continue_classified_objective_index(bind(held),hi,old_package=oldpkg,new_package=newpkg)
    added=tmp_path/'added.jsonl';added.write_text('new admitted graph fixture\n')
    ae=[{**entries[0],'id':'added','binding':'f'*64}];av={**value,'key':{**value['key'],'source_sha256':bind(added)['sha256']},'entries':ae,'metadata_sha256':c.digest(ae)}
    ai=index_ref(tmp_path/'added.pkl',av);ap=c.continue_classified_objective_index(bind(added),ai,old_package=oldpkg,new_package=newpkg)
    train=put(tmp_path/'new-train.json',{'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs',**bind(filtered)},{'kind':'native-graphs',**bind(added)}]})
    def proof(train,parts,hparts):return {'schema':m.SCHEMA,'train':train,'train_sha256':train['sha256'],'held':bind(held),
        'indexes':parts,'held_indexes':hparts,'model_allocations':0,'optimizer_updates':0,'student_pointer_writes':0,'graph_position_expansions':0}
    prior=proof(oldtrain,[oldpart],[hp]);oldproof=put(tmp_path/'old-proof.json',prior)
    current=proof(train,[fp,ap],[heldpart]);current['objective_policy_continuation']={'schema':'natlang.native-classified-objective-continuation/1',
        'prior_proof':oldproof,'old_package':str(oldpkg),'new_package':str(newpkg),'withdrawals':[{'prior_source_sha256':bind(source)['sha256'],
        'new_source':bind(filtered),'filter_proof':filt,'withdrawal_proof':wd}]}
    policy=put(tmp_path/'ce.json',declared(digest=train['sha256']))
    old={'native_graphs':{**oldtrain,'binding':'oldtrain'},'held_native_graphs':{**bind(held),'binding':'held'},
         'source_classification_migration':{**oldproof,'binding':'oldproof'}}
    def save():
        cp=put(tmp_path/'new-proof.json',current)
        return {**old,'native_graphs':{**train,'binding':'newtrain'},'continued_native_graphs':old['native_graphs'],
            'source_classification_migration':{**cp,'binding':'newproof'},'withdrawn_native_graphs':{**wd,'binding':'withdrawal'},
            'ce_supervision_policy':{**policy,'binding':'ce'}}
    return old,save,current,prior,put,index_ref


def test_actual_combined_recipe_ce_data_rekey_withdrawal(combined,tmp_path):
    old,save,current,prior,*_=combined;new=save()
    assert recipe._ce_supervision_input_continuation(old,new)
    def plan(roles):return {'recipe':{'stages':[{'id':'full','kind':'native_state_sft','inputs':{k:v['binding']for k,v in roles.items()}}]},'stage_inputs':{'full':roles}}
    changes=recipe.unstarted_stage_input_changes(plan(old),plan(new),tmp_path,started_stage_ids=('full',))
    assert changes['full']['kind']=='ce_supervision_continuation'
    assert prior['indexes'][0]['rows']==3 and current['indexes'][0]['rows']==2


@pytest.mark.parametrize('fault',['missing_extension','prior','undeclared_deletion','held','source','class','binding','policy','withdrawal','undeclared_withdrawal','manifest'])
def test_rejects_source_class_held_and_withdrawal_tampering(combined,fault):
    old,save,current,prior,put,index_ref=combined
    if fault=='missing_extension':current.pop('objective_policy_continuation')
    elif fault=='prior':current['objective_policy_continuation']['prior_proof']['sha256']='0'*64
    elif fault=='undeclared_deletion':current['indexes'].pop(0)
    elif fault=='held':current['held_indexes']=[]
    elif fault=='source':current['indexes'][0]['source']=current['indexes'][1]['source']
    elif fault in ('class','binding','policy'):
        part=current['indexes'][0];p=Path(part['new_index']['path']);v=pickle.loads(p.read_bytes())
        if fault=='class':v['entries'][0].update(communication=False,producer_consumer=False)
        elif fault=='binding':v['entries'][0]['binding']='9'*64
        else:v['key']['policy_sha256']='0'*64
        v['metadata_sha256']=c.digest(v['entries']);part['new_index']=index_ref(p,v)
    elif fault=='withdrawal':
        w=current['objective_policy_continuation']['withdrawals'][0];p=Path(w['filter_proof']['path']);v=json.loads(p.read_text());v['new_rows']+=1;w['filter_proof']=put(p,v)
    elif fault=='undeclared_withdrawal':current['objective_policy_continuation']['withdrawals']=[]
    else:
        p=Path(current['train']['path']);v=json.loads(p.read_text());v['sources'].pop();ref=put(p,v);current['train']=ref;current['train_sha256']=ref['sha256']
    new=save()
    # The role must match a changed exact manifest in this mutation too.
    if fault=='manifest':new['native_graphs'].update(current['train'])
    try:
        allowed=recipe._ce_supervision_input_continuation(old,new)
    except (ValueError,KeyError):
        return
    assert allowed is False


@pytest.mark.parametrize('fault',['raw_class_flip','added_unclassified','old_policy_files','producer_path'])
def test_rejects_unwitnessed_new_raw_classes_and_rekey_authority(combined,fault,tmp_path):
    old,save,current,prior,put,index_ref=combined
    part=current['indexes'][-1]
    if fault=='raw_class_flip':
        path=Path(part['new_index']['path']);v=pickle.loads(path.read_bytes());v['entries'][0].update(communication=True,producer_consumer=True)
        v['metadata_sha256']=c.digest(v['entries']);part['new_index']=index_ref(path,v)
    elif fault=='added_unclassified':
        path=Path(part['old_index']['path']);v=pickle.loads(path.read_bytes())
        for e in v['entries']:
            for key in m.CLASS_FIELDS:e.pop(key,None)
        v['metadata_sha256']=c.digest(v['entries']);part['old_index']=index_ref(path,v)
    elif fault=='old_policy_files':
        part['old_policy']['common/jsonio.py']=part['old_policy']['common/hashing.py']
    else:
        p=tmp_path/'same-classifier.py';p.write_bytes(Path(part['producer']['path']).read_bytes());part['producer']=m._binding(p)
    with pytest.raises(ValueError):recipe._ce_supervision_input_continuation(old,save())


def test_recipe_and_publication_recursively_bind_actual_proof_dependencies(combined,tmp_path):
    from natlang_neuralese.train.native_classification_continuation import classification_proof_dependencies
    from scripts.coreweave.provider_dependencies import dependency_closure
    old,save,current,prior,put,index_ref=combined;new=save();ref=new['source_classification_migration']
    dependencies=classification_proof_dependencies(ref)
    paths={x['path']for x in dependencies}
    extra=current['objective_policy_continuation'];wd=extra['withdrawals'][0]
    wanted={ref['path'],extra['prior_proof']['path'],wd['filter_proof']['path'],wd['withdrawal_proof']['path'],
            str(Path(extra['new_package'])/'data/native_sources.py'),str(Path(extra['old_package'])/'data/native_sources.py'),
            str(Path(recipe.__file__).parent/'native_classification_continuation.py'),str(Path(c.__file__))}
    assert wanted<=paths
    bound=recipe._bind_native_source_dependencies(copy.deepcopy(new))
    assert {x['path']for x in bound['source_classification_migration']['dependencies']}==paths
    cfg=put(tmp_path/'config.json',{'source_classification_migration':ref,
        'trainer_parameters':{'source_classification_migration':ref['path']}})
    checkout=Path(__file__).resolve().parents[2]
    # A composed runtime can live outside the source checkout under test.
    runtime_checkout = Path(c.__file__).resolve().parents[2]
    closure=dependency_closure({'config':cfg},namespace=tmp_path,
        readonly_input_roots=sorted({str(checkout), str(runtime_checkout)}))
    actual={x['path']for x in closure['files']+closure['assets']}
    assert wanted<=actual
    Path(wd['filter_proof']['path']).write_text('tampered filter')
    with pytest.raises(ValueError):classification_proof_dependencies(ref)


def test_dependency_enumeration_authenticates_shared_files_once_per_call(combined, monkeypatch):
    from collections import Counter
    from natlang_neuralese.train.native_classification_continuation import classification_proof_dependencies
    _, _, _, prior, put, _ = combined
    part = prior['indexes'][0]
    ref = put(Path(part['source']['path']).parent / 'repeated-proof.json',
              {'schema': m.SCHEMA, 'indexes': [part] * 100, 'held_indexes': []})
    wanted = [ref, part['source'], part['old_index'], part['new_index'],
              part['producer'], *part['old_policy'].values()]
    expected = list({item['path']: m._binding(item['path']) for item in wanted}.values())
    calls = Counter()
    checked = m._checked
    def check(value):
        calls[value['path']] += 1
        return checked(value)
    monkeypatch.setattr(m, '_checked', check)
    assert classification_proof_dependencies(ref) == expected
    assert calls == Counter({item['path']: 1 for item in expected})
    assert classification_proof_dependencies(ref) == expected
    assert calls == Counter({item['path']: 2 for item in expected})


def test_dependency_enumeration_stats_unique_files_not_repeated_parts(combined, monkeypatch):
    from collections import Counter
    from natlang_neuralese.train import native_classification_continuation as continuation
    _, _, _, prior, put, _ = combined
    part = prior['indexes'][0]
    ref = put(Path(part['source']['path']).parent / 'many-repeated-parts.json',
              {'schema': m.SCHEMA, 'indexes': [part] * 1000, 'held_indexes': []})
    signatures = Counter()
    original = continuation._index_signature
    def signature(path):
        signatures[str(path)] += 1
        return original(path)
    monkeypatch.setattr(continuation, '_index_signature', signature)
    result = continuation.classification_proof_dependencies(ref)
    # Authenticate before/after first consumption and check again at return.
    # Duplicate declarations must not turn this into thousands of filesystem calls.
    assert signatures == Counter({item['path']: 3 for item in result})


@pytest.mark.parametrize('field,value', [('sha256', '0' * 64), ('bytes', 1)])
def test_dependency_reuse_rejects_contradictory_occurrence(combined, field, value):
    from natlang_neuralese.train.native_classification_continuation import classification_proof_dependencies
    _, _, _, prior, put, _ = combined
    part = copy.deepcopy(prior['indexes'][0])
    # Both index references name the same actual file in this historical proof.
    part['new_index'] = {**part['new_index'], field: value}
    ref = put(Path(part['source']['path']).parent / 'conflicting-proof.json',
              {'schema': m.SCHEMA, 'indexes': [part], 'held_indexes': []})
    with pytest.raises(ValueError):
        classification_proof_dependencies(ref)


@pytest.mark.parametrize('fault', ['same-mtime', 'replacement', 'symlink-retarget'])
def test_dependency_reuse_checks_all_files_again_before_return(combined, monkeypatch, fault):
    import os
    from natlang_neuralese.train.native_classification_continuation import classification_proof_dependencies
    _, _, _, prior, put, _ = combined
    part = copy.deepcopy(prior['indexes'][0])
    source = Path(part['source']['path'])
    if fault == 'symlink-retarget':
        link = source.with_name('source-alias.jsonl')
        link.symlink_to(source)
        part['source'] = m._binding(link)
    ref = put(source.parent / 'mutation-proof.json',
              {'schema': m.SCHEMA, 'indexes': [part], 'held_indexes': []})
    checked = m._checked
    def check(value):
        result = checked(value)
        if value['path'] == part['producer']['path']:
            raw = source.read_bytes()
            state = source.stat()
            if fault == 'same-mtime':
                source.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
                os.utime(source, ns=(state.st_atime_ns, state.st_mtime_ns))
            else:
                replacement = source.with_name('different-inode.jsonl')
                replacement.write_bytes(raw)
                os.utime(replacement, ns=(state.st_atime_ns, state.st_mtime_ns))
                if fault == 'replacement':
                    replacement.replace(source)
                else:
                    link.unlink()
                    link.symlink_to(replacement)
        return result
    monkeypatch.setattr(m, '_checked', check)
    with pytest.raises(ValueError, match='changed during enumeration'):
        classification_proof_dependencies(ref)


def test_historical_manifest_child_byte_annotation_is_not_intrinsic_authority(tmp_path):
    from natlang_neuralese.train.native_classification_continuation import _leaves
    source=tmp_path/'child.jsonl';source.write_text('immutable native graph source\n');ref=m._binding(source)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**ref,'bytes':916}]}))
    top=m._binding(manifest)
    assert source.stat().st_size!=916
    assert _leaves(top)=={(ref['sha256'],'native-graphs',None)}
    with pytest.raises(ValueError):_leaves({**top,'bytes':top['bytes']+1})
    source.write_text('changed source still rejects even when bytes annotations ignored\n')
    with pytest.raises(ValueError):_leaves(top)


def test_rekey_cached_source_hash_reuse_and_same_mtime_mutation_rejection(case,monkeypatch):
    import builtins
    import os
    old,new,source,bind,entries,value,write,index=case
    source_ref=bind(source);index_ref=write()
    first=c.continue_classified_objective_index(source_ref,index_ref,old_package=old,new_package=new)
    original=builtins.open
    def guarded(path,*args,**kwargs):
        if Path(path).resolve()==source.resolve():raise AssertionError('unchanged source body rehashed')
        return original(path,*args,**kwargs)
    with monkeypatch.context()as scoped:
        scoped.setattr(builtins,'open',guarded)
        assert c.continue_classified_objective_index(source_ref,index_ref,old_package=old,new_package=new)==first
    state=source.stat();raw=source.read_bytes();source.write_bytes(bytes([raw[0]^1])+raw[1:]);os.utime(source,ns=(state.st_atime_ns,state.st_mtime_ns))
    with pytest.raises(ValueError):c.continue_classified_objective_index(source_ref,index_ref,old_package=old,new_package=new)


def test_shared_policy_maps_are_authenticated_twice_per_call(combined, monkeypatch):
    from natlang_neuralese.train import native_classification_continuation as continuation
    old,save,current,prior,*_=combined;new=save()
    paths={ref['path'] for ref in current['indexes'][0]['old_policy'].values()}
    calls=[];original=m._checked
    def checked(ref):
        if ref['path'] in paths:calls.append(ref['path'])
        return original(ref)
    monkeypatch.setattr(m,'_checked',checked)
    for invocation in range(2):
        assert continuation.verify_objective_proof_extension(old,new,prior,current)
        assert all(calls.count(path)==2*(invocation+1) for path in paths)


@pytest.mark.parametrize('fault', ['bytes', 'sha256'])
def test_reused_policy_map_does_not_hide_changed_reference(combined, fault):
    from natlang_neuralese.train import native_classification_continuation as continuation
    old,save,current,prior,*_=combined;new=save()
    # An independently copied later source map must retain all its authority.
    refs=current['held_indexes'][0]['old_policy']=copy.deepcopy(current['held_indexes'][0]['old_policy'])
    ref=next(iter(refs.values()))
    ref[fault]=ref['bytes']+1 if fault=='bytes' else '0'*64
    with pytest.raises(ValueError,match='artifact changed'):
        continuation.verify_objective_proof_extension(old,new,prior,current)


def test_policy_mutation_during_source_inventory_fails_final_recheck(combined, monkeypatch):
    import os
    from natlang_neuralese.train import native_classification_continuation as continuation
    old,save,current,prior,*_=combined;new=save()
    target=Path(next(iter(current['indexes'][0]['old_policy'].values()))['path'])
    original=continuation._leaves;mutated=False
    def leaves(ref):
        nonlocal mutated
        result=original(ref)
        if not mutated:
            state=target.stat();raw=target.read_bytes()
            target.write_bytes(bytes([raw[0]^1])+raw[1:])
            os.utime(target,ns=(state.st_atime_ns,state.st_mtime_ns));mutated=True
        return result
    monkeypatch.setattr(continuation,'_leaves',leaves)
    with pytest.raises(ValueError,match='artifact changed'):
        continuation.verify_objective_proof_extension(old,new,prior,current)
    assert mutated


def test_shallow_dependency_check_preserves_current_parts_scope(combined, tmp_path):
    from natlang_neuralese.train.native_classification_continuation import classification_proof_dependencies
    _, _, proof, _, put, _ = combined
    proof = copy.deepcopy(proof)
    missing = {'path': str(tmp_path / 'independent-predecessor-check.json'), 'sha256': '0' * 64}
    proof['objective_policy_continuation']['prior_proof'] = missing
    ref = put(tmp_path / 'current-parts-only.json', proof)
    actual = classification_proof_dependencies(ref, follow_continuations=False)
    expected = {ref['path']}
    for part in proof['indexes'] + proof['held_indexes']:
        expected.update(part[key]['path'] for key in ('source', 'old_index', 'new_index', 'producer'))
        expected.update(value['path'] for value in part['old_policy'].values())
    assert {item['path'] for item in actual} == expected
    with pytest.raises((ValueError, OSError)):
        classification_proof_dependencies(ref)
