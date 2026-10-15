"""render_parts.merge_parts keeps global split decontamination across parts (Pop 2026-10-12)."""
import json
import hashlib
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'neuralese_data'))
import render_parts  # noqa: E402


def write(root, part, tok, rows):
    path = root / 'parts' / part / 'program-header' / tok
    path.mkdir(parents=True, exist_ok=True)
    (path / 'text-data.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_completed_part_compression_preserves_exact_bytes_and_resumes(tmp_path):
    rows = [{'id': 'a', 'split': 'train', 'source_groups': ['g'], 'text': 'native context'}]
    write(tmp_path, 'p00', 'lfm25', rows)
    plain = tmp_path / 'parts/p00/program-header/lfm25/text-data.jsonl'
    before = plain.read_bytes()
    render_parts.compress_part_views(tmp_path / 'parts/p00')
    render_parts.compress_part_views(tmp_path / 'parts/p00')
    paths = render_parts.part_files(tmp_path, 'program-header', 'lfm25')
    assert len(paths) == 1 and paths[0].suffix == '.zst' and not plain.exists()
    assert b''.join(line for line, _ in render_parts.read_rows(paths[0])) == before


def test_render_part_forwards_additive_protection_and_source_hold_inputs(tmp_path, monkeypatch):
    part = tmp_path / 'parts' / 'p00'
    part.mkdir(parents=True)
    (tmp_path / '.p00.turns.jsonl').write_text('{}\n')
    commands = []
    monkeypatch.setattr(render_parts, 'wait_for_disk', lambda _: None)
    monkeypatch.setattr(render_parts, 'pending_units', lambda _: [])
    monkeypatch.setattr(render_parts.time, 'sleep', lambda _: None)

    def run(command):
        commands.append(command)
        (part / 'receipt.json').write_text('{}')
        return type('Result', (), {'returncode': 0})()

    monkeypatch.setattr(render_parts.subprocess, 'run', run)
    render_parts.render_part([], tmp_path, 0, 0, 1, 'fixture', 'fixture', 1, 1, 1,
                             protected=Path('/custom/protected.json'),
                             reserved=Path('/custom/reserved.json'),
                             held_text=[Path('/compact/global-held-groups.jsonl')],
                             decision_exclude_ids=[Path('/holds/decision-ids.jsonl')],
                             exclude_programs=[Path('/holds/program-ids.jsonl')])
    assert len(commands) == 1
    command = commands[0]
    held = command[command.index('--held-text') + 1:command.index('--decision-exclude-ids')]
    assert held == [str(render_parts.REPO / path) for path in render_parts.HELD_TEXT] + [
        '/compact/global-held-groups.jsonl']
    assert command[command.index('--protected') + 1] == '/custom/protected.json'
    assert command[command.index('--reserved') + 1] == '/custom/reserved.json'
    assert command[command.index('--decision-exclude-ids') + 1] == '/holds/decision-ids.jsonl'
    assert command[command.index('--exclude-programs') + 1] == '/holds/program-ids.jsonl'


def test_legacy_default_resume_is_labeled_and_cannot_claim_new_policy(tmp_path):
    (tmp_path / 'receipt.json').write_text(json.dumps({'parts': 1}))
    receipt = render_parts.render([], tmp_path, 'fixture', 'fixture')
    assert receipt['legacy_policy_unbound'] is True
    assert 'rechecked at union' in receipt['legacy_policy_note']
    with pytest.raises(RuntimeError, match='fresh output'):
        render_parts.render([], tmp_path, 'fixture', 'fixture', held_text=[Path('/new/held.jsonl')])


def test_policy_binding_rejects_changed_inputs(tmp_path):
    path = tmp_path / 'render-policy-binding.json'
    first = {'source_sha256': 'a' * 64, 'controls': ['base', 'new']}
    assert render_parts.validate_binding(path, first, legacy_description='test output')
    assert render_parts.validate_binding(path, first, legacy_description='test output')
    with pytest.raises(RuntimeError, match='different render policy/input identity'):
        render_parts.validate_binding(path, {**first, 'controls': ['base']}, legacy_description='test output')


def test_compact_held_index_proof_is_part_of_render_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(render_parts, 'HELD_TEXT', [])
    protected = tmp_path / 'protected.json'
    reserved = tmp_path / 'reserved.json'
    held = tmp_path / 'held.jsonl'
    proof = held.with_suffix('.proof.json')
    for path, contents in ((protected, '{}'), (reserved, '{}'), (held, '{}\n'), (proof, '{"source":"a"}')):
        path.write_text(contents)
    kwargs = dict(sources=[], corpus_id='fixture', cpus=1, budget_gb=1, docker_gb=1,
                  part_decisions=10, compress_all=False, protected=protected, reserved=reserved,
                  held_text=[held], decision_exclude_ids=[], exclude_programs=[])
    first = render_parts.render_binding(**kwargs)
    assert any(row['kind'] == 'control-provenance-proof' and row['path'] == str(proof)
               for row in first['controls'])
    proof.write_text('{"source":"b"}')
    second = render_parts.render_binding(**kwargs)
    assert first != second


@pytest.mark.parametrize('compress_all', [False, True])
def test_cross_part_held_group_duplicate_and_twin_alignment(tmp_path, compress_all):
    for tok in ('lfm25', 'mellum21'):
        write(tmp_path, 'p00', tok, [
            {'id': 'a', 'split': 'train', 'source_groups': ['g1'], 'text': 'A' + tok},
            {'id': 'b', 'split': 'train', 'source_groups': ['g2'], 'text': 'B' + tok}])
        write(tmp_path, 'p01', tok, [
            {'id': 'c', 'split': 'test', 'source_groups': ['g1'], 'text': 'C' + tok},
            {'id': 'd', 'split': 'train', 'source_groups': ['g3'], 'text': 'B' + tok},
            {'id': 'e', 'split': 'train', 'source_groups': ['g4'], 'text': ('E' + tok) if tok == 'lfm25' else ('B' + tok)}])
    merged = render_parts.merge_parts(tmp_path, compress_all=compress_all)
    filename = 'text-data.jsonl.zst' if compress_all else 'text-data.jsonl'
    ids = {tok: [row['id'] for _, row in render_parts.read_rows(tmp_path / 'program-header' / tok / filename)]
           for tok in ('lfm25', 'mellum21')}
    # a: shares held group g1 with test doc c of another part; d: duplicate of b; e: duplicate only for mellum21,
    # so it is dropped for both twins.
    assert ids == {'lfm25': ['b', 'c'], 'mellum21': ['b', 'c']}
    assert merged['merged']['program-header/lfm25']['merge_omissions'] == {
        'exact_duplicate_document_across_parts': 2, 'train_document_shares_held_source_group_across_parts': 1}
    assert not list((tmp_path / 'parts').glob('*/*/*/text-data.jsonl'))
    for tok in ('lfm25', 'mellum21'):
        path = tmp_path / 'program-header' / tok / filename
        binding = merged['merged'][f'program-header/{tok}']
        assert binding['text_data_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        if compress_all:
            decoded = b''.join(line for line, _ in render_parts.read_rows(path))
            assert binding['decoded_text_data_sha256'] == hashlib.sha256(decoded).hexdigest()
            assert binding['decoded_text_data_bytes'] == len(decoded)


@pytest.mark.parametrize('whole_trajectory', [False, True])
def test_native_shard_merge_compression_keeps_held_closure_and_exact_bytes(tmp_path, whole_trajectory):
    import importlib.util
    from natlang_neuralese.common.jsonio import write_jsonl_text, open_jsonl_text
    spec=importlib.util.spec_from_file_location('native_stream_renderer',Path(__file__).resolve().parents[2]/'scripts/neuralese_data/render_native_turn_corpus.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    rows=[{'id':'held','split':'test','source_groups':['protected'],'text':'héld','token_ids':[1,2],'supervised_suffix_start':1},
          {'id':'leak','split':'train','source_groups':['protected'],'text':'other','token_ids':[2],'supervised_suffix_start':0},
          {'id':'good','split':'train','source_groups':['good'],'text':'你好','token_ids':[3,4],'supervised_suffix_start':1}]
    if whole_trajectory:
        for row in rows:
            row.pop('supervised_suffix_start')
    outputs=[]
    for compressed in (False,True):
        root=tmp_path/str(compressed);src=root/'work/program-header/fixture';src.mkdir(parents=True)
        for i,row in enumerate(rows):
            path=src/(f'{i}.jsonl.zst' if i%2 else f'{i}.jsonl')
            with write_jsonl_text(path) as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        result=module.merge(root,'program-header','fixture',['0','1','2'],set(),lambda ids: (__import__('collections').Counter(assistant_reply=len(ids)), [5]*len(ids)),compress_text=compressed)
        target=root/'program-header/fixture'/result.get('text_data_file','text-data.jsonl')
        with open_jsonl_text(target) as stream:decoded=stream.read().encode()
        outputs.append(decoded)
        assert result['documents']==2 and result['merge_omissions']=={'train_document_shares_held_source_group':1}
        if compressed:
            assert result['decoded_text_data_sha256']==hashlib.sha256(decoded).hexdigest()
            assert result['decoded_text_data_bytes']==len(decoded)
    assert outputs[0]==outputs[1]
