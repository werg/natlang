"""Whole pending rows reuse measured capacity without changing their disposition."""
import json
import hashlib
import pickle
from pathlib import Path
import pytest
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.native_capacity_activation import SCHEMA
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.common.hashing import sha256_file_hex
from test_native_resident import ordinary_coverage_part


def fixture(tmp_path, monkeypatch, *, filtered=False, role_interpretation='ordinary-text'):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    source, part = ordinary_coverage_part(tmp_path, 'natlang.acquired-native-capacity-partition/1')
    coverage = Path(part['coverage']['path'])
    rows = [json.loads(v) for v in coverage.read_text().splitlines()]
    for row in rows: row['partition'] = row['disposition'] = 'remaining-long'
    coverage.write_text(''.join(json.dumps(v)+'\n' for v in rows))
    receipt_path = Path(part['receipt']['path']); receipt = json.loads(receipt_path.read_text())
    receipt['manifests']['fixture:train:remaining-long'] = receipt['manifests'].pop('fixture:train:fit')
    receipt['counts']['fixture']['train:remaining-long'] = receipt['counts']['fixture'].pop('train:fit')
    receipt['files'] = [{**resident.binding(receipt_path.parent/v['path']), 'path':v['path'],
                         'bytes':(receipt_path.parent/v['path']).stat().st_size} for v in receipt['files']]
    write(receipt_path, receipt)
    package = Path(resident.__file__).parents[1]; code=resident.code_inventory(package)
    evidence=tmp_path/'capacity-evidence.json';write(evidence, {'fixture':True})
    certificate=tmp_path/'capacity.json';write(certificate, {
        'schema':'natlang.native-resident-geometry-qualification/1','passed':True,'identity':{},
        'runtime_code_sha256':resident.digest(code),'evidence':resident.binding(evidence),
        'limits':{'forward_tokens':3 if filtered else 4,'kv_slots':4}})
    context=tmp_path/'current.json';payload={'geometry_qualification':resident.binding(certificate),'geometry_identity':{},'plan':{'code':code}};write(context, {'schema':resident.SCHEMA,'generation':resident.digest(payload),'payload':payload})
    part.update(schema=SCHEMA, receipt=resident.binding(receipt_path),coverage=resident.binding(coverage),
                original_source=part['source'], resident_current=resident.binding(context))
    if filtered:
        from scripts.neuralese_data.withdraw_native_graph_targets import filter_source, continue_filtered_index
        leaf=source.children[0].path
        graphs=[json.loads(v) for v in leaf.read_text().splitlines()]
        for value in graphs:value['conversion']={'role_interpretation':role_interpretation}
        # Rebuild the original receipt/index for these explicitly ordinary bytes.
        leaf.write_text(''.join(json.dumps(v)+'\n'for v in graphs))
        manifest=source.path;doc=json.loads(manifest.read_text());doc['sources'][0].update(resident.binding(leaf));write(manifest,doc)
        receipt['files']=[{**resident.binding(receipt_path.parent/v['path']),'path':v['path'],
                          'bytes':(receipt_path.parent/v['path']).stat().st_size}for v in receipt['files']]
        write(receipt_path,receipt);part.update(receipt=resident.binding(receipt_path),original_source=resident.binding(manifest))
        source=type(source)(manifest,'tokenizer-sha',required_split='train')
        cache=next(path for path in (tmp_path/'cache').glob('*.pkl')
                   if pickle.loads(path.read_bytes())['key']['source_sha256']==sha256_file_hex(leaf))
        cached=pickle.loads(cache.read_bytes());output=tmp_path/'selected.jsonl'
        proof=filter_source(leaf,output,[{**source.entries[1],'source_sha256':sha256_file_hex(leaf)}],source_sha256=sha256_file_hex(leaf))
        continue_filtered_index(cache,proof,index_sha256=sha256_file_hex(cache),metadata_sha256=cached['metadata_sha256'])
        proof_path=tmp_path/'filter.json';write(proof_path,proof)
        part.update(source=resident.binding(output),filter_proof=resident.binding(proof_path),
                    original_index={'cache':resident.binding(cache),'metadata_sha256':cached['metadata_sha256']})
        source=type(source)(output,'tokenizer-sha',required_split='train')
    original_manifest=Path(part['original_source']['path'])
    leaf_ref=json.loads(original_manifest.read_text())['sources'][0]
    original_leaf=Path(leaf_ref['path'])
    if not original_leaf.is_absolute():original_leaf=original_manifest.parent/original_leaf
    raw_rows=original_leaf.read_bytes().splitlines(keepends=True)
    original_sha=sha256_file_hex(original_leaf);original_path='/original/fixture/train/native-graphs.jsonl'
    covered=[json.loads(v)for v in coverage.read_text().splitlines()]
    for index,(row,raw_row)in enumerate(zip(covered,raw_rows,strict=True)):
        row.update(source={'path':original_path,'sha256':original_sha,'row_index':index},
                   source_graph_sha256=resident.digest(json.loads(raw_row)),
                   source_row_bytes_sha256=hashlib.sha256(raw_row).hexdigest())
    coverage.write_text(''.join(json.dumps(v)+'\n'for v in covered));part['coverage']=resident.binding(coverage)
    receipt['source_bindings']=[{'path':original_path,'sha256':original_sha,'split':'train','rows':len(raw_rows)}]
    receipt['files']=[{**resident.binding(receipt_path.parent/v['path']),'path':v['path'],
                      'bytes':(receipt_path.parent/v['path']).stat().st_size}for v in receipt['files']]
    producer=Path(__file__).parents[2]/'scripts/neuralese_data/partition_acquired_native_capacity.py'
    audit_path=tmp_path/'audit.json';samples_path=tmp_path/'render-samples.json';proof_path=tmp_path/'row-proof.jsonl'
    proof_path.write_text(''.join(json.dumps({'model':row['model'],'id':row['id'],'split':'train',
        'graph_sha256':row['source_graph_sha256'],'tokens':row['geometry']['forward_tokens']})+'\n'for row in covered))
    write(audit_path,{'schema':'natlang.acquired-session-admission-audit/1','status':'passed',
        'row_proof_sha256':sha256_file_hex(proof_path),'conversions':[{'outputs':receipt['source_bindings']}]})
    write(samples_path,{'passed':True,'cohort_audit':resident.binding(audit_path)})
    receipt['binding']={'verifier_sha256':sha256_file_hex(producer),
        'audit':resident.binding(audit_path),'source_render_samples':resident.binding(samples_path)}
    part.update(partition_verifier=resident.binding(producer),admission_audit=resident.binding(audit_path),
        render_samples=resident.binding(samples_path),row_proof=resident.binding(proof_path))
    write(receipt_path,receipt);part['receipt']=resident.binding(receipt_path)
    registered=tmp_path/'registered.json';write(registered,{'schema':'natlang.corpus-snapshot/1','id':'fixture',
        'files':[*receipt['files'],*[{**resident.binding(v),'path':v.name,'bytes':v.stat().st_size}for v in(audit_path,samples_path,proof_path)],{**resident.binding(receipt_path),'path':receipt_path.name,'bytes':receipt_path.stat().st_size}]})
    part.update(registered_manifest=resident.binding(registered),corpus_root=str(tmp_path))
    from natlang_neuralese.train.native_pending_sources import declaration
    pending=tmp_path/'pending.json';declared=declaration(original_manifest,original_manifest,
        tokenizer_fingerprint='tokenizer-sha',role_ids=())
    write(pending,{**declared,'tokenizer_fingerprint':'tokenizer-sha'})
    part['pending_obligations']=resident.binding(pending)
    return source,part,package


@pytest.mark.parametrize('filtered',[False,True])
def test_actual_whole_delta_report_accepts_original_remaining_long_and_exact_filtered_view(tmp_path,monkeypatch,filtered):
    source,part,package=fixture(tmp_path,monkeypatch,filtered=filtered)
    report=tmp_path/'report.json'
    ref=resident.prepare_whole_delta_geometry(source,[part],report,package=package)
    value=json.loads(report.read_text())
    assert value['new_train_rows']==(1 if filtered else 2)
    assert resident.geometry_from_evidence(source,0,ref,package=package)=={'forward_tokens':3 if filtered else 4,'kv_slots':3 if filtered else 4}
    receipt=json.loads(Path(part['receipt']['path']).read_text())
    assert 'fixture:train:fit'not in receipt['manifests']
    assert Path(part['original_source']['path']).exists()


@pytest.mark.parametrize('bad',['coverage-id','coverage-kv','unbound-member','false-runtime','failed-certificate','missing-filter'])
def test_activation_rejects_changed_original_geometry_or_authority_before_report(tmp_path,monkeypatch,bad):
    source,part,package=fixture(tmp_path,monkeypatch,filtered=bad=='missing-filter')
    if bad=='missing-filter':del part['filter_proof']
    elif bad.startswith('coverage'):
        path=Path(part['coverage']['path']);rows=[json.loads(v)for v in path.read_text().splitlines()]
        if bad=='coverage-id':rows[0]['id']='forged'
        else:rows[0]['geometry']['kv_slots']+=1
        path.write_text(''.join(json.dumps(v)+'\n'for v in rows));part['coverage']=resident.binding(path)
        receipt_path=Path(part['receipt']['path']);receipt=json.loads(receipt_path.read_text())
        receipt['files']=[{**v,**{k:part['coverage'][k]for k in('sha256',)}}if v['path']==path.name else v for v in receipt['files']]
        for v in receipt['files']:
            if v['path']==path.name:v['bytes']=path.stat().st_size
        write(receipt_path,receipt);part['receipt']=resident.binding(receipt_path)
    elif bad=='unbound-member':part['coverage']={**part['coverage'],'sha256':'0'*64}
    else:
        context_path=Path(part['resident_current']['path']);context=json.loads(context_path.read_text())
        if bad=='false-runtime':context['payload']['plan']['code']={}
        else:
            path=Path(context['payload']['geometry_qualification']['path']);certificate=json.loads(path.read_text());certificate['passed']=False;write(path,certificate);context['payload']['geometry_qualification']=resident.binding(path)
        context['generation']=resident.digest(context['payload']);write(context_path,context);part['resident_current']=resident.binding(context_path)
    output=tmp_path/'report.json'
    with pytest.raises((ValueError,KeyError)):
        resident.prepare_whole_delta_geometry(source,[part],output,package=package)
    assert not output.exists()


def test_filter_cannot_change_raw_whitespace_even_when_graph_and_index_bindings_match(tmp_path,monkeypatch):
    import hashlib
    source,part,package=fixture(tmp_path,monkeypatch,filtered=True)
    selected=source.path
    raw=selected.read_bytes();selected.write_bytes(b' '+raw)
    proof_path=Path(part['filter_proof']['path']);proof=json.loads(proof_path.read_text())
    proof['output']={**resident.binding(selected),'bytes':selected.stat().st_size}
    proof['retained_raw_sha256']=hashlib.sha256(selected.read_bytes()).hexdigest()
    write(proof_path,proof);part['filter_proof']=resident.binding(proof_path);part['source']=resident.binding(selected)
    source=type(source)(selected,'tokenizer-sha',required_split='train')
    with pytest.raises(ValueError,match='retained original row bytes'):
        resident.prepare_whole_delta_geometry(source,[part],tmp_path/'report.json',package=package)
    assert not(tmp_path/'report.json').exists()


def test_qualifier_frontier_does_not_allow_omitting_another_fitting_row(tmp_path,monkeypatch):
    source,part,package=fixture(tmp_path,monkeypatch)
    leaf=tmp_path/'only-one.jsonl';leaf.write_text(Path(source.children[0].path).read_text().splitlines()[0]+'\n')
    selected=type(source)(leaf,'tokenizer-sha',required_split='train');part['source']=resident.binding(leaf)
    with pytest.raises(ValueError,match='all and only fitting whole rows'):
        resident.prepare_whole_delta_geometry(selected,[part],tmp_path/'report.json',package=package)
    assert not(tmp_path/'report.json').exists()


@pytest.mark.parametrize('field,value',[('tokenizer','wrong'),('policy_sha256','0'*64),
    ('pair_model','mellum21'),('role_ids',[4,3,2,1]),('kind','semantic-pairs'),('split','test')])
def test_filtered_original_index_requires_exact_runtime_parser_key(tmp_path,monkeypatch,field,value):
    source,part,package=fixture(tmp_path,monkeypatch,filtered=True)
    original=Path(part['original_index']['cache']['path'])
    cached=pickle.loads(original.read_bytes());cached['key'][field]=value
    forged=tmp_path/'forged-index.pkl';forged.write_bytes(pickle.dumps(cached))
    part['original_index']['cache']=resident.binding(forged)
    with pytest.raises(ValueError,match='original index differs'):
        resident.prepare_whole_delta_geometry(source,[part],tmp_path/'report.json',package=package)
    assert not(tmp_path/'report.json').exists()


@pytest.mark.parametrize('bad',['raw-hash','graph-hash','source-sha','duplicate-source-position','source-position-oob','unregistered-receipt'])
def test_registered_source_coverage_requires_original_source_positions_and_row_hashes(tmp_path,monkeypatch,bad):
    source,part,package=fixture(tmp_path,monkeypatch,filtered=True)
    coverage=Path(part['coverage']['path']);rows=[json.loads(v)for v in coverage.read_text().splitlines()]
    if bad=='raw-hash':rows[0]['source_row_bytes_sha256']='0'*64
    elif bad=='graph-hash':rows[0]['source_graph_sha256']='0'*64
    elif bad=='source-sha':rows[0]['source']['sha256']='0'*64
    elif bad=='duplicate-source-position':rows[1]['source']['row_index']=rows[0]['source']['row_index']
    elif bad=='source-position-oob':rows[0]['source']['row_index']=999
    coverage.write_text(''.join(json.dumps(v)+'\n'for v in rows));part['coverage']=resident.binding(coverage)
    receipt_path=Path(part['receipt']['path']);receipt=json.loads(receipt_path.read_text())
    if bad=='unregistered-receipt':receipt['unregistered_note']='changed after corpus registration'
    receipt['files']=[{**resident.binding(receipt_path.parent/v['path']),'path':v['path'],
        'bytes':(receipt_path.parent/v['path']).stat().st_size}for v in receipt['files']]
    write(receipt_path,receipt);part['receipt']=resident.binding(receipt_path)
    registered=Path(part['registered_manifest']['path'])
    if bad!='unregistered-receipt':
        document=json.loads(registered.read_text());updates={v['path']:v for v in receipt['files']}
        updates[receipt_path.name]={**resident.binding(receipt_path),'path':receipt_path.name,'bytes':receipt_path.stat().st_size}
        document['files']=[updates.get(v['path'],v)for v in document['files']];write(registered,document)
        part['registered_manifest']=resident.binding(registered)
    output=tmp_path/'report.json'
    with pytest.raises(ValueError):resident.prepare_whole_delta_geometry(source,[part],output,package=package)
    assert not output.exists()


def test_activation_keeps_over_frontier_rows_in_executable_pending_declaration(tmp_path,monkeypatch):
    source,part,package=fixture(tmp_path,monkeypatch,filtered=True)
    path=Path(part['pending_obligations']['path']);value=json.loads(path.read_text())
    assert len(value['sources'][0]['rows'])==2
    value['sources'][0]['rows']=value['sources'][0]['rows'][:1];write(path,value)
    part['pending_obligations']=resident.binding(path)
    with pytest.raises(ValueError,match='preserve all original whole pending'):
        resident.prepare_whole_delta_geometry(source,[part],tmp_path/'report.json',package=package)
    assert not(tmp_path/'report.json').exists()


@pytest.mark.parametrize('field',['partition_verifier','admission_audit','render_samples','row_proof'])
def test_activation_requires_original_ordinary_producer_chain(tmp_path,monkeypatch,field):
    source,part,package=fixture(tmp_path,monkeypatch)
    part[field]={**part[field],'sha256':'0'*64}
    with pytest.raises(ValueError):resident.prepare_whole_delta_geometry(source,[part],tmp_path/'bad.json',package=package)


def test_activation_filter_proof_cannot_name_another_output(tmp_path,monkeypatch):
    source,part,package=fixture(tmp_path,monkeypatch,filtered=True)
    path=Path(part['filter_proof']['path']);proof=json.loads(path.read_text())
    proof['output']['path']=str(tmp_path/'another.jsonl');write(path,proof);part['filter_proof']=resident.binding(path)
    with pytest.raises(ValueError,match='filter output differs'):
        resident.prepare_whole_delta_geometry(source,[part],tmp_path/'bad.json',package=package)


@pytest.mark.parametrize('bad',['proof-geometry','proof-graph','preference-audit','different-producer'])
def test_unfiltered_activation_rejects_resealed_nonordinary_producer_evidence(tmp_path,monkeypatch,bad):
    source,part,package=fixture(tmp_path,monkeypatch)
    audit_path=Path(part['admission_audit']['path']);audit=json.loads(audit_path.read_text())
    proof_path=Path(part['row_proof']['path']);proofs=[json.loads(v)for v in proof_path.read_text().splitlines()]
    if bad=='proof-geometry':proofs[0]['tokens']+=1
    if bad=='proof-graph':proofs[0]['graph_sha256']='0'*64
    if bad.startswith('proof-'):
        proof_path.write_text(''.join(json.dumps(v)+'\n'for v in proofs));audit['row_proof_sha256']=sha256_file_hex(proof_path)
    if bad=='preference-audit':audit['schema']='natlang.acquired-offline-preference-admission/1'
    write(audit_path,audit);part['admission_audit']=resident.binding(audit_path);part['row_proof']=resident.binding(proof_path)
    samples_path=Path(part['render_samples']['path']);samples=json.loads(samples_path.read_text());samples['cohort_audit']=part['admission_audit'];write(samples_path,samples);part['render_samples']=resident.binding(samples_path)
    receipt_path=Path(part['receipt']['path']);receipt=json.loads(receipt_path.read_text());receipt['binding']['audit']=part['admission_audit'];receipt['binding']['source_render_samples']=part['render_samples']
    if bad=='different-producer':receipt['binding']['verifier_sha256']='0'*64
    write(receipt_path,receipt);part['receipt']=resident.binding(receipt_path)
    registered=Path(part['registered_manifest']['path']);document=json.loads(registered.read_text())
    refs={v.name:{**resident.binding(v),'path':v.name,'bytes':v.stat().st_size}for v in(audit_path,proof_path,samples_path,receipt_path)}
    document['files']=[refs.get(v['path'],v)for v in document['files']];write(registered,document);part['registered_manifest']=resident.binding(registered)
    with pytest.raises(ValueError,match='producer/audit chain|all-row admission proof'):
        resident.prepare_whole_delta_geometry(source,[part],tmp_path/'bad.json',package=package)


def test_filtered_whole_trajectory_chat_role_graphs_preserve_original_weights_and_bindings(tmp_path,monkeypatch):
    source,part,package=fixture(tmp_path,monkeypatch,filtered=True,role_interpretation='whole-trajectory-chat-roles')
    result=resident.prepare_whole_delta_geometry(source,[part],tmp_path/'report.json',package=package)
    value=json.loads(Path(result['path']).read_text());assert value['new_train_rows']==1
    proof=json.loads(Path(part['filter_proof']['path']).read_text());assert proof['old_rows']==2 and proof['new_rows']==1
    assert resident.geometry_from_evidence(source,0,result,package=package)=={'forward_tokens':3,'kv_slots':3}


@pytest.mark.parametrize('receipt_matches',[True,False])
def test_multiple_reviewed_producers_still_require_exact_receipted_producer(tmp_path,monkeypatch,receipt_matches):
    from natlang_neuralese.train import native_capacity_activation as activation
    source,part,package=fixture(tmp_path,monkeypatch)
    producer=tmp_path/'reviewed-historical-producer.py'
    producer.write_text('# separately reviewed ordinary producer fixture\n')
    producer_ref=resident.binding(producer)
    monkeypatch.setattr(activation,'_ORDINARY_PARTITION_VERIFIERS',
        activation._ORDINARY_PARTITION_VERIFIERS|{producer_ref['sha256']})
    part['partition_verifier']=producer_ref
    receipt_path=Path(part['receipt']['path']);receipt=json.loads(receipt_path.read_text())
    if receipt_matches:receipt['binding']['verifier_sha256']=producer_ref['sha256']
    write(receipt_path,receipt);part['receipt']=resident.binding(receipt_path)
    manifest_path=Path(part['registered_manifest']['path']);manifest=json.loads(manifest_path.read_text())
    manifest['files']=[{**part['receipt'],'path':receipt_path.name,'bytes':receipt_path.stat().st_size}
        if entry['path']==receipt_path.name else entry for entry in manifest['files']]
    write(manifest_path,manifest);part['registered_manifest']=resident.binding(manifest_path)
    if not receipt_matches:
        with pytest.raises(ValueError,match='producer/audit chain'):
            resident.prepare_whole_delta_geometry(source,[part],tmp_path/'bad.json',package=package)
    else:
        result=resident.prepare_whole_delta_geometry(source,[part],tmp_path/'report.json',package=package)
        assert resident.geometry_from_evidence(source,0,result,package=package)=={'forward_tokens':4,'kv_slots':4}


def represented_fixture(tmp_path, monkeypatch, *, mixed=False):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from scripts.neuralese_data.withdraw_native_graph_targets import filter_source, continue_filtered_index
    if mixed:
        original_builder = ordinary_coverage_part
        def three_rows(root, kind):
            source, part = original_builder(root, kind)
            leaf = source.children[0].path
            graphs = [json.loads(line) for line in leaf.read_text().splitlines()]
            third = json.loads(json.dumps(graphs[0])); third['id'] = 'third'
            graphs.append(third)
            leaf.write_text(''.join(json.dumps(row)+'\n' for row in graphs))
            manifest = json.loads(source.path.read_text())
            manifest['sources'][0].update(resident.binding(leaf)); write(source.path, manifest)
            source = type(source)(source.path, 'tokenizer-sha', required_split='train')
            coverage = Path(part['coverage']['path'])
            coverage.write_text(''.join(json.dumps({'id':entry['id'], 'model':'fixture', 'split':'train',
                'partition':'fit', 'disposition':'fit', 'partition_row_index':index,
                'geometry':{'forward_tokens':entry['forward_tokens'], 'kv_slots':entry['forward_tokens']}})+'\n'
                for index, entry in enumerate(source.entries)))
            receipt_path = Path(part['receipt']['path']); receipt = json.loads(receipt_path.read_text())
            receipt['counts']['fixture']['train:fit']['rows'] = 3
            receipt['files'] = [{**resident.binding(receipt_path.parent/v['path']), 'path':v['path'],
                'bytes':(receipt_path.parent/v['path']).stat().st_size} for v in receipt['files']]
            write(receipt_path, receipt)
            part.update(source=resident.binding(source.path), coverage=resident.binding(coverage), receipt=resident.binding(receipt_path))
            return source, part
        monkeypatch.setitem(globals(), 'ordinary_coverage_part', three_rows)
    _, part, package = fixture(tmp_path, monkeypatch, filtered=True)
    context_path = Path(part['resident_current']['path'])
    context = json.loads(context_path.read_text())
    certificate_path = Path(context['payload']['geometry_qualification']['path'])
    certificate = json.loads(certificate_path.read_text())
    certificate['limits'] = {'forward_tokens':3 if mixed else 4, 'kv_slots':4}
    write(certificate_path, certificate)
    context['payload']['geometry_qualification'] = resident.binding(certificate_path)
    original = NativeGraphSource(part['original_source']['path'], 'tokenizer-sha', required_split='train')
    leaf = original.children[0].path
    prior_path = tmp_path/'prior.jsonl'
    prior_path.write_bytes(leaf.read_bytes().splitlines(keepends=True)[0])
    prior = NativeGraphSource(prior_path, 'tokenizer-sha', required_split='train')
    context['payload']['train'] = resident.binding(prior_path)
    context['generation'] = resident.digest(context['payload'])
    write(context_path, context)
    part['resident_current'] = resident.binding(context_path)
    descriptor_path = tmp_path/'represented.json'
    descriptor = {'schema':'natlang.native-capacity-represented-prefix/1',
        'prior_generation':context['generation'], 'prior_train':context['payload']['train'],
        'indexes':[{'source':resident.binding(prior_path), 'kind':'native-graphs',
                    'witness':resident._source_witness(prior, package)[0]}]}
    write(descriptor_path, descriptor)
    part['represented_prefix'] = resident.binding(descriptor_path)
    selected_path = tmp_path/'missing.jsonl'
    proof = filter_source(leaf, selected_path, [{**entry, 'source_sha256':sha256_file_hex(leaf)}
                          for entry in original.entries[:2 if mixed else 1]],
                          source_sha256=sha256_file_hex(leaf))
    witness = part['original_index']
    continue_filtered_index(witness['cache']['path'], proof, index_sha256=witness['cache']['sha256'],
                            metadata_sha256=witness['metadata_sha256'])
    proof_path = tmp_path/'represented-filter.json'
    write(proof_path, proof)
    part.update(source=resident.binding(selected_path), filter_proof=resident.binding(proof_path))
    selected = NativeGraphSource(selected_path, 'tokenizer-sha', required_split='train')
    return selected, part, package, descriptor_path


def test_exact_prior_index_can_exclude_represented_whole_row_without_losing_pending(tmp_path, monkeypatch):
    selected, part, package, _ = represented_fixture(tmp_path, monkeypatch)
    from natlang_neuralese.train.native_capacity_activation import geometry_part
    from natlang_neuralese.train.native_source_preparation import NativeSourcePreparation
    actual, maxima, provenance, count = geometry_part(part,
        preparation=NativeSourcePreparation('tokenizer-sha', ()), package=package)
    assert actual.entries == selected.entries and count == 1
    assert maxima['forward_tokens']['forward_tokens'] == 4
    assert part['represented_prefix'] in provenance
    from natlang_neuralese.train.native_pending_sources import load_pending_sources
    pending = load_pending_sources(part['pending_obligations']['path'], tokenizer_fingerprint='tokenizer-sha')
    assert len(pending['rows']) == 2


@pytest.mark.parametrize('bad', ['generation', 'train', 'unrelated-leaf', 'split', 'policy',
                                  'missing-index', 'corrupt-cache', 'binding', 'class'])
def test_represented_exclusion_needs_exact_current_source_and_indexed_row(tmp_path, monkeypatch, bad):
    selected, part, package, descriptor_path = represented_fixture(tmp_path, monkeypatch)
    descriptor = json.loads(descriptor_path.read_text())
    item = descriptor['indexes'][0]
    if bad == 'generation':
        descriptor['prior_generation'] = '0'*64
    elif bad == 'train':
        descriptor['prior_train'] = part['original_source']
    elif bad == 'unrelated-leaf':
        item['source'] = part['source']
    elif bad in ('split', 'policy'):
        item['witness']['key']['split' if bad == 'split' else 'policy_sha256'] = 'test' if bad == 'split' else '0'*64
    elif bad == 'missing-index':
        descriptor['indexes'] = []
    else:
        path = Path(item['witness']['cache']['path'])
        if bad == 'corrupt-cache':
            path.write_bytes(b'corrupted')
        else:
            cached = pickle.loads(path.read_bytes())
            cached['entries'][0]['binding' if bad == 'binding' else 'communication'] = '0'*64 if bad == 'binding' else True
            cached['metadata_sha256'] = resident.digest(cached['entries'])
            path.write_bytes(pickle.dumps(cached))
            item['witness'].update(cache=resident.binding(path), metadata_sha256=cached['metadata_sha256'])
    write(descriptor_path, descriptor)
    part['represented_prefix'] = resident.binding(descriptor_path)
    with pytest.raises(ValueError):
        resident.prepare_whole_delta_geometry(selected, [part], tmp_path/'bad.json', package=package)
    assert not(tmp_path/'bad.json').exists()


def test_represented_and_over_frontier_exclusions_preserve_both_original_obligations(tmp_path, monkeypatch):
    selected, part, package, _ = represented_fixture(tmp_path, monkeypatch, mixed=True)
    output = tmp_path/'mixed.json'
    resident.prepare_whole_delta_geometry(selected, [part], output, package=package)
    assert [entry['id'] for entry in selected.entries] == ['third']
    assert json.loads(output.read_text())['new_train_rows'] == 1
    from natlang_neuralese.train.native_pending_sources import load_pending_sources
    pending = load_pending_sources(part['pending_obligations']['path'], tokenizer_fingerprint='tokenizer-sha')
    assert len(pending['rows']) == 3
