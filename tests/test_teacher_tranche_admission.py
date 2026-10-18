import json
from pathlib import Path

import pytest

from scripts.coreweave.teacher_tranche_admission import (
    Tranche, admitted_predecessors, file_binding, packet_references, preceding_tranches,
    materialization_partition,
)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return file_binding(path)


def packet(tmp_path):
    cases = tmp_path / 'cases.jsonl'
    cases.write_text('{"id":"original","split":"train"}\n')
    cases_ref = file_binding(cases)
    qualification = put(tmp_path / 'qualification.json', {
        'schema': 'natlang.existing-native-reference-qualification/1',
        'passed': True, 'training_admission': False, 'model_calls': 0,
        'qualified_cases_binding': cases_ref,
    })
    selection = {'cases_sha256': cases_ref['sha256'], 'training_admission': False, 'cases': 1}
    references = {
        'selection': put(tmp_path / 'selection.json', selection),
        'selected_cases': cases_ref,
        'global_protection_join': put(tmp_path / 'protection.json', {'passed': True}),
    }
    prepared = put(tmp_path / 'prepared.json', {
        'cases': cases_ref, 'qualification': qualification, 'references': references,
    })
    ready = put(tmp_path / 'ready.json', {'entries': [{'qualified_packet': prepared}]})
    collection = {'source_cases_sha256': cases_ref['sha256'], 'protected_selection': selection,
                  'source_provenance': {key: {**value, 'path': '/remote/' + key}
                                        for key, value in references.items()}}
    return ready, collection, prepared, references


def test_batches_have_distinct_zero_names_and_preserve_old_names(tmp_path):
    old, new = Tranche(21, 0), Tranche(22, 0)
    assert old.corpus_id.endswith('recovery7853-tranche0000')
    assert new.corpus_id.endswith('recovery7853-batch0022-tranche0000')
    assert old.archive_name == 'teacher-batch-0021-tranche-0000'
    assert new.archive_name == 'teacher-batch-0022-tranche-0000'
    assert old.admission(tmp_path) == tmp_path / 'admission/tranche-0000'
    assert new.admission(tmp_path) == tmp_path / 'admission/batch-0022/tranche-0000'
    assert new.admission(tmp_path, recovery=True) == tmp_path / 'admission-recovery/batch-0022/tranche-0000'


@pytest.mark.parametrize('batch,number', [(True, 0), (20, 0), (22, -1), (22, False)])
def test_namespace_rejects_invalid_typed_numbers(batch, number):
    with pytest.raises(ValueError):
        Tranche(batch, number)


def test_resolves_actual_packet_and_keeps_remote_provenance(tmp_path):
    ready, collection, prepared, references = packet(tmp_path)
    result = packet_references(ready, collection)
    assert result['prepared'] == prepared
    assert result['references'] == references
    assert result['remote_references']['selection']['path'] == '/remote/selection'


def test_embedded_selection_is_typed_exact(tmp_path):
    ready, collection, *_ = packet(tmp_path)
    collection['protected_selection']['cases'] = 1.0
    with pytest.raises(ValueError, match='selected tasks differ'):
        packet_references(ready, collection)


def test_ambiguous_and_wrong_packet_rejected(tmp_path):
    ready, collection, prepared, _ = packet(tmp_path)
    ready = put(Path(ready['path']), {'entries': [{'qualified_packet': prepared}] * 2})
    with pytest.raises(ValueError, match='exactly one'):
        packet_references(ready, collection)
    collection['source_cases_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='exactly one'):
        packet_references(ready, collection)


def test_changed_cases_and_remote_selection_hash_reject(tmp_path):
    ready, collection, _, references = packet(tmp_path)
    collection['source_provenance']['selection']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='provenance differs'):
        packet_references(ready, collection)
    Path(references['selected_cases']['path']).write_text('different\n')
    with pytest.raises(ValueError, match='cases changed'):
        packet_references(ready, collection)


def corpus(root, batch, number):
    cid = Tranche(batch, number).corpus_id
    folder = root / 'runs' / cid
    outputs = {}
    files = []
    for model in ('lfm25', 'mellum21'):
        path = folder / model / 'train/native-graphs.jsonl.zst'
        path.parent.mkdir(parents=True)
        path.write_bytes((model + cid).encode())
        outputs[model] = {'outputs': {'train': file_binding(path)}}
        files.append({**file_binding(path), 'path': str(path.relative_to(folder))})
    ref = put(folder / 'validation-proof.json', {
        'status': 'passed', 'training_admission': True, 'models': outputs,
    })
    files.append({**ref, 'path': 'validation-proof.json'})
    put(root / 'training/corpus-manifests' / (cid + '.json'), {
        'id': cid, 'path': str(folder.relative_to(root)), 'files': files,
    })
    return {'id': cid, 'path': str(folder.relative_to(root)), 'training_admission': True}


def test_predecessors_span_batches_in_registry_order_exclude_self(tmp_path):
    first = corpus(tmp_path, 21, 27)
    second = corpus(tmp_path, 22, 0)
    put(tmp_path / 'training/neuralese_corpora.json', {'corpora': [first, second]})
    assert [row['id'] for row in admitted_predecessors(tmp_path, second['id'])] == [first['id']]
    assert [row['id'] for row in admitted_predecessors(tmp_path, Tranche(22, 1).corpus_id)] == [first['id'], second['id']]


def test_predecessor_requires_manifest_and_both_exact_admitted_views(tmp_path):
    entry = corpus(tmp_path, 21, 27)
    put(tmp_path / 'training/neuralese_corpora.json', {'corpora': [entry]})
    (tmp_path / entry['path'] / 'mellum21/train/native-graphs.jsonl.zst').write_bytes(b'changed')
    with pytest.raises(ValueError, match='graph differs'):
        admitted_predecessors(tmp_path, Tranche(22, 0).corpus_id)


def test_unregistered_output_is_not_dedup_authority(tmp_path):
    corpus(tmp_path, 22, 0)
    put(tmp_path / 'training/neuralese_corpora.json', {'corpora': []})
    assert admitted_predecessors(tmp_path, Tranche(22, 1).corpus_id) == []
    with pytest.raises(FileNotFoundError):
        preceding_tranches(tmp_path, tmp_path / 'spool', Tranche(22, 1))


def test_exact_rejected_predecessor_retains_raw_outcomes(tmp_path):
    spool = tmp_path / 'spool'
    put(tmp_path / 'training/neuralese_corpora.json', {'corpora': []})
    prior = Tranche(22, 0)
    directory = prior.admission(spool)
    directory.mkdir(parents=True)
    source = directory / 'native-trajectories.jsonl'
    source.write_bytes(b'')
    output = prior.admission(spool, recovery=True) / 'restored/outputs/failed.jsonl'
    output.parent.mkdir(parents=True)
    output.write_text('{"outcome":"failed"}\n')
    proof_path = directory / 'canonical-source-verification.json'
    proof = {'status': 'passed', 'training_admission': False, 'rows': 0, 'selected_outputs': 1,
             'source': file_binding(source), 'rejected_outputs': [{
                 'key': 'original', 'reason': 'exact failure', 'source_output': 'outputs/failed.jsonl',
                 'sha256': file_binding(output)['sha256'],
             }]}
    put(proof_path, proof)
    admitted, rejected = preceding_tranches(tmp_path, spool, Tranche(22, 1))
    assert admitted == []
    assert rejected[0]['batch'] == 22 and rejected[0]['number'] == 0
    proof['rejected_outputs'][0]['sha256'] = '0' * 64
    put(proof_path, proof)
    with pytest.raises(ValueError, match='preserved source'):
        preceding_tranches(tmp_path, spool, Tranche(22, 1))


def partition_fixture():
    raw = {'good': {'trajectory': [None, None]}, 'held': {'trajectory': [None, None, None]}}
    reason = 'shared native materializer did not qualify this original decision'
    omitted = [{'source_row_id': 'good', 'teacher_turn_index': 1, 'reason': 'failed proposal'}]
    omitted += [{'source_row_id': 'held', 'teacher_turn_index': index, 'reason': reason,
                 'decision': {'reason': reason}} for index in range(3)]
    return raw, omitted


def test_partial_batch_preserves_good_graphs_and_all_repair_actions():
    raw, omitted = partition_fixture()
    proof = materialization_partition(raw, [('good', 0), ('good', 1)], omitted, [('good', 0)])
    assert proof == dict(source_rows=2, materialized_source_rows=1, repair_source_rows=['held'],
                        repair_source_actions=3, source_actions=5, emitted_actions=1,
                        deduplicated_actions=0, other_omitted_actions=1, training_admission=False)


def test_held_source_can_never_supply_graph_or_lose_action():
    raw, omitted = partition_fixture()
    with pytest.raises(ValueError, match='conserve'):
        materialization_partition(raw, [('good', 0), ('good', 1)], omitted[1:], [('good', 0)])
    with pytest.raises(ValueError, match='conserve'):
        materialization_partition(raw, [('good', 0), ('good', 1)], omitted, [('good', 0), ('held', 0)])


def test_partition_rejects_partial_materializer_row_and_duplicate_omission():
    raw, omitted = partition_fixture()
    with pytest.raises(ValueError, match='partially lost'):
        materialization_partition(raw, [('good', 0)], omitted, [('good', 0)])
    with pytest.raises(ValueError, match='duplicate'):
        materialization_partition(raw, [('good', 0), ('good', 1)], omitted + omitted[-1:], [('good', 0)])


def test_held_omission_reason_and_types_cannot_be_rewritten():
    raw, omitted = partition_fixture()
    omitted[-1]['decision']['reason'] = 'approved'
    with pytest.raises(ValueError, match='exact materializer omission'):
        materialization_partition(raw, [('good', 0), ('good', 1)], omitted, [('good', 0)])


def test_deduplicated_action_is_counted_once_without_a_second_graph():
    raw, omitted = partition_fixture()
    proof = materialization_partition(raw, [('good', 0), ('good', 1)], omitted, [], [('good', 0)])
    assert proof['emitted_actions'] == 0 and proof['deduplicated_actions'] == 1


def test_repair_attempt_preserves_archive_keys_but_separates_all_mutable_custody(tmp_path):
    ordinary, repair, later = Tranche(21, 0), Tranche(21, 0, 1), Tranche(21, 0, 2)
    assert ordinary.archive_name == repair.archive_name == later.archive_name
    assert len({x.corpus_id for x in (ordinary, repair, later)}) == 3
    assert repair.corpus_id.endswith('batch0021-repair0001-tranche0000')
    assert repair.exports(tmp_path) == tmp_path / 'repair-exports/batch-0021-attempt-0001'
    assert ordinary.exports(tmp_path) == tmp_path / 'batch-0021'
    for recovery in (False, True):
        assert len({x.admission(tmp_path, recovery=recovery) for x in (ordinary, repair, later)}) == 3
    assert repair.admission(tmp_path) == tmp_path / 'admission/repair-attempts/batch-0021/attempt-0001/tranche-0000'


@pytest.mark.parametrize('attempt', [0, -1, True, 1.0, '1'])
def test_repair_attempt_is_typed_positive_integer(attempt):
    with pytest.raises(ValueError, match='repair attempt'):
        Tranche(21, 0, attempt)


def test_original_admitted_tranche_cannot_discharge_repair_attempt_predecessor(tmp_path):
    ordinary = corpus(tmp_path, 21, 0)
    put(tmp_path / 'training/neuralese_corpora.json', {'corpora': [ordinary]})
    with pytest.raises(FileNotFoundError):
        preceding_tranches(tmp_path, tmp_path / 'spool', Tranche(21, 1, 1))
    assert admitted_predecessors(tmp_path, Tranche(21, 0, 1).corpus_id)[0]['id'] == ordinary['id']


@pytest.mark.parametrize('attempt',[None,1])
def test_quality_view_preserves_actual_source_exports_recovery_and_attempt(tmp_path,attempt):
    original=Tranche(23,0,attempt);view=Tranche(23,0,attempt,'source-argument-oracle-v1')
    assert view.exports(tmp_path)==original.exports(tmp_path)
    assert view.archive_name==original.archive_name
    assert view.admission(tmp_path,recovery=True)==original.admission(tmp_path,recovery=True)
    assert view.admission(tmp_path)!=original.admission(tmp_path)
    assert view.admission(tmp_path).is_relative_to(tmp_path/'admission-quality-views/source-argument-oracle-v1')
    assert view.corpus_id==original.corpus_id+'-quality-source-argument-oracle-v1'
    assert view.attempt==original.attempt


@pytest.mark.parametrize('name',['','../bad','a/b','..','.','A','a--b',True,1,[]])
def test_quality_view_name_cannot_escape_or_fake_typed_identity(name):
    with pytest.raises(ValueError,match='quality view'):Tranche(23,0,quality_view=name)


def source_phase_fixture(tmp_path):
    from scripts.coreweave.teacher_tranche_admission import completed_source_phase
    source_dir = tmp_path / 'positive'
    source_dir.mkdir()
    source = source_dir / 'native-trajectories.jsonl'
    source.write_text(json.dumps({'id': 'row', 'outcome': {'status': 'done', 'accepted': True,
        'oracle': {'accepted': True}, 'checks': {'files': True}}}) + '\n')
    source_ref = file_binding(source)
    raw = put(tmp_path / 'restored/raw.json', {'original': 'unchanged'})
    manifest = put(tmp_path / 'restored/raw.manifest.json', {'completed': 1})
    context = put(tmp_path / 'restored/context/packet.json', {'source': 'train'})
    put(tmp_path / 'restored/tranche-manifest.json', {
        'context': {'packet.json': context}, 'outputs': [{'output': 'raw.json',
            'output_sha256': raw['sha256'], 'output_bytes': raw['bytes'],
            'output_manifest': 'raw.manifest.json', 'manifest_sha256': manifest['sha256'],
            'manifest_bytes': manifest['bytes']}],
    })
    private = put(tmp_path / 'private.json', {'identity': {'sha256': 'archive-sha', 'bytes': 123},
        'independent_verification': {'sha256': 'archive-sha', 'bytes': 123}})
    recovery = put(tmp_path / 'recovery.json', {'restoration': {'status': 'passed'},
        'archive_identity': {'sha256': 'archive-sha', 'bytes': 123}, 'private_receipt': private['path'],
        'restored': str(tmp_path / 'restored')})
    operator = put(tmp_path / 'frozen-operator.json', {'mechanism': 'historical'})
    inventory = put(tmp_path / 'operators.json', {'helper': operator, 'operators': [operator]})
    execution = put(tmp_path / 'execution.json', {'inventory': inventory, 'pipeline': operator})
    put(source_dir / 'canonical-source-verification.json', {
        'schema': 'natlang.teacher-live-tranche-canonical-source-verification/1',
        'status': 'passed', 'training_admission': False, 'rows': 1, 'selected_outputs': 1,
        'source': source_ref})
    put(source_dir / 'transfer-receipt.json', {
        'schema': 'natlang.verified-teacher-trajectory-transfer/1', 'training_admission': False,
        'combined': {**source_ref, 'rows': 1}, 'files': [raw],
        'private_export': {'archive_sha256': 'archive-sha'}})
    put(source_dir / 'batch-source-bindings.json', {'batches': ['original-packet']})
    receipt = put(tmp_path / 'completed-source.json',
                  completed_source_phase(source_dir, recovery, execution))
    return source_dir, receipt


def test_completed_source_resumes_without_rewriting_historical_runtime_metadata(tmp_path):
    from scripts.coreweave.teacher_tranche_admission import verify_completed_source_phase
    directory, reference = source_phase_fixture(tmp_path)
    before = (directory / 'native-trajectories.jsonl').read_bytes()
    # A current runtime outside the frozen source phase changes independently.
    put(tmp_path / 'current-runtime.json', {'version': 'new-checker'})
    proof = verify_completed_source_phase(reference, directory)
    assert proof['source_rows'] == 1 and proof['training_admission'] is False
    assert proof['requires_fresh_conversion_and_admission'] is True
    assert (directory / 'native-trajectories.jsonl').read_bytes() == before


@pytest.mark.parametrize('relative', [
    'positive/native-trajectories.jsonl', 'positive/transfer-receipt.json',
    'positive/batch-source-bindings.json', 'positive/canonical-source-verification.json',
    'restored/raw.json', 'restored/raw.manifest.json', 'restored/context/packet.json',
    'restored/tranche-manifest.json', 'frozen-operator.json', 'operators.json',
    'execution.json', 'private.json', 'recovery.json'])
def test_completed_source_rejects_any_changed_original_custody(tmp_path, relative):
    from scripts.coreweave.teacher_tranche_admission import verify_completed_source_phase
    directory, reference = source_phase_fixture(tmp_path)
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        verify_completed_source_phase(reference, directory)


def test_completed_source_needs_exact_receipt_and_directory(tmp_path):
    from scripts.coreweave.teacher_tranche_admission import verify_completed_source_phase
    directory, reference = source_phase_fixture(tmp_path)
    receipt = json.loads(Path(reference['path']).read_bytes())
    receipt['requires_fresh_conversion_and_admission'] = False
    tampered = put(tmp_path / 'bad-completion.json', receipt)
    with pytest.raises(ValueError, match='receipt differs'):
        verify_completed_source_phase(tampered, directory)
    with pytest.raises(FileNotFoundError):
        verify_completed_source_phase(reference, tmp_path / 'another-source')


def test_completed_source_detects_same_size_mtime_restored_mid_verification(tmp_path, monkeypatch):
    import os
    import scripts.coreweave.teacher_tranche_admission as module
    directory, reference = source_phase_fixture(tmp_path)
    original = module.file_binding
    target = directory / 'native-trajectories.jsonl'
    changed = False
    def binding(path):
        nonlocal changed
        result = original(path)
        if Path(path).name == 'frozen-operator.json' and not changed:
            changed = True
            st = target.stat()
            target.write_bytes(target.read_bytes().replace(b'row', b'bad'))
            os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
        return result
    monkeypatch.setattr(module, 'file_binding', binding)
    with pytest.raises(ValueError, match='changed during verification'):
        module.verify_completed_source_phase(reference, directory)


def test_partial_parent_decisions_never_count_as_whole_accepted_sources():
    raw, omitted = partition_fixture()
    proof = materialization_partition(raw, [('good',0),('good',1)], omitted, [('good',0)],
                                      partial_parent_rows=['good'])
    assert proof['materialized_source_rows'] == 0
    assert proof['partial_source_rows'] == ['good']
    assert proof['partial_source_actions'] == 2
    assert proof['partial_parent_tasks_qualified'] is False
    assert proof['source_actions'] == proof['emitted_actions'] + proof['other_omitted_actions'] + proof['repair_source_actions']
    with pytest.raises(ValueError, match='absent from materialized'):
        materialization_partition(raw, [('good',0),('good',1)], omitted, [('good',0)],
                                  partial_parent_rows=['held'])


def test_default_materializer_continuation_requires_all_decision_bytes_and_typed_summary(tmp_path):
    from scripts.coreweave.teacher_tranche_admission import verify_default_materializer_continuation
    root=tmp_path/'runtime'; root.mkdir(); asset=root/'materializer.js'; asset.write_text('current implementation')
    source=tmp_path/'source.jsonl'; source.write_text('{"original":true}\n')
    source_ref=file_binding(source)
    summary={'output':'decisions.jsonl','eligible_decisions':1,'rejected_rows':2,'accepted_rows':1}
    refs=[]
    for name in ('old','current'):
        folder=tmp_path/name; folder.mkdir(); output=folder/'decisions.jsonl'; output.write_text('{"id":"original-target"}\n')
        receipt={'schema':'natlang.verified-teacher-decision-materialization/1',
                 'identity':{'source_sha256':source_ref['sha256'],'direct_answers':True,
                             'runtime':{'materializer.js':file_binding(asset)['sha256']}},
                 'output_sha256':file_binding(output)['sha256'],'summary':summary}
        put(folder/'receipt.json',receipt); refs.append(file_binding(folder/'receipt.json'))
    proof=verify_default_materializer_continuation(*refs,source_reference=source_ref,current_repository=root)
    assert proof['decision_bytes_identical'] is True and proof['training_admission'] is False
    old_bytes=Path(refs[0]['path']).read_bytes()
    def check():
        return verify_default_materializer_continuation(refs[0],file_binding(Path(refs[1]['path'])),
            source_reference=source_ref,current_repository=root)
    current=Path(refs[1]['path']); original=json.loads(current.read_bytes())
    for mutation in ('summary','summary_type','review','source','output','runtime'):
        value=json.loads(json.dumps(original))
        if mutation=='summary':value['summary']['rejected_rows']=3
        elif mutation=='summary_type':value['summary']['eligible_decisions']=1.0
        elif mutation=='review':value['identity']['decision_review']={'sha256':'a'*64}
        elif mutation=='source':value['identity']['source_sha256']='a'*64
        elif mutation=='output':value['output_sha256']='a'*64
        elif mutation=='runtime':value['identity']['runtime']['materializer.js']='a'*64
        put(current,value)
        with pytest.raises(ValueError):check()
    put(current,original)
    (current.parent/'decisions.jsonl').write_text('{"id":"different-target"}\n')
    value=json.loads(current.read_bytes());value['output_sha256']=file_binding(current.parent/'decisions.jsonl')['sha256'];put(current,value)
    with pytest.raises(ValueError,match='every original decision'):check()
    assert Path(refs[0]['path']).read_bytes()==old_bytes


def test_requalified_identical_cases_select_actual_full_packet_authority(tmp_path):
    ready, collection, prepared, references = packet(tmp_path)
    old = json.loads(Path(prepared['path']).read_text())
    old['references'] = dict(old['references'])
    old['references']['selection'] = put(tmp_path / 'old-selection.json', {
        **collection['protected_selection'], 'reference_epoch': 'older',
    })
    old_prepared = put(tmp_path / 'old-prepared.json', old)
    for entries in ([old_prepared, prepared], [prepared, old_prepared]):
        current_ready = put(Path(ready['path']), {'entries': [
            {'qualified_packet': ref} for ref in entries]})
        result = packet_references(current_ready, collection)
        assert result['prepared'] == prepared
        assert result['references'] == references
    # Same cases and same embedded selection still cannot substitute a different
    # authenticated provenance file for the actually captured collection.
    old['references']['selection'] = references['selection']
    old['references']['global_protection_join'] = put(tmp_path / 'old-join.json', {
        'passed': True, 'reference_epoch': 'older',
    })
    old_prepared = put(tmp_path / 'old-prepared.json', old)
    current_ready = put(Path(ready['path']), {'entries': [
        {'qualified_packet': old_prepared}, {'qualified_packet': prepared}]})
    assert packet_references(current_ready, collection)['prepared'] == prepared
    collection['source_provenance']['selection']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='provenance differs'):
        packet_references(current_ready, collection)


def test_guided_reference_root_omissions_conserve_sampled_children():
    import copy
    import pytest
    from scripts.coreweave.teacher_tranche_admission import materialization_partition
    root = {'kind': 'authored_reference_eval', 'source': 'curriculum.reference.root',
            'source_program_id': 'program', 'code_sha256': 'a'*64, 'sampled': False, 'status': 'finished'}
    provenance = {k:v for k,v in root.items() if k != 'status'}
    provenance['kind'] = 'authored_reference_root_eval'
    raw = {'row': {'task': {'program_ir': {'id': 'program'}},
        'collection_guidance': {'training_admission': False, 'root_action': root},
        'trajectory': [{'action_provenance': provenance}, {'assistant': {'content': 'correct child'}}]}}
    omission = {'source_row_id': 'row', 'teacher_turn_index': 0,
        'reason': 'shared native materializer did not qualify this original decision',
        'decision': {'reason': 'shared native materializer did not qualify this original decision'}}
    proof = materialization_partition(raw, {('row',1)}, [omission], [('row',1)])
    assert proof['source_actions'] == 2 and proof['emitted_actions'] == 1
    assert proof['authored_static_root_actions'] == 1 and proof['guided_source_rows'] == ['row']
    assert proof['materialized_source_rows'] == 0 and proof['guided_parent_traces_admitted'] is False
    for mutation in ('sampled', 'program', 'code', 'guidance', 'missing_child', 'invented_omission'):
        changed = copy.deepcopy(raw); omissions = copy.deepcopy([omission]); keys = {('row',1)}
        if mutation == 'sampled':changed['row']['trajectory'][0]['action_provenance']['sampled'] = True
        elif mutation == 'program':changed['row']['trajectory'][0]['action_provenance']['source_program_id'] = 'other'
        elif mutation == 'code':changed['row']['trajectory'][0]['action_provenance']['code_sha256'] = 'b'*64
        elif mutation == 'guidance':changed['row']['collection_guidance']['training_admission'] = True
        elif mutation == 'missing_child':changed['row']['trajectory'].append({'assistant': {'content': 'lost'}})
        elif mutation == 'invented_omission':omissions[0]['reason'] = 'generic archive'
        with pytest.raises(ValueError):materialization_partition(changed, keys, omissions, [('row',1)])
    with pytest.raises(ValueError, match='static roots'):
        materialization_partition(raw, {('row',0),('row',1)}, [], [('row',0),('row',1)])
