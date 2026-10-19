import hashlib
import pytest
from scripts.neuralese_data.build_static_repository_graphs import source_row, disposition
from scripts.neuralese_data.splits import close_group_sets


def original(body='const name = "é";\n', **changes):
    value = dict(owner='Owner', repo='Package', relpath='src/main.js', commit_sha='1'*40,
        text='<REPO_NAME>Owner/Package <FILE_PATH>src/main.js\n'+body,
        size_bytes=len(body.encode()), sha256=hashlib.sha256(body.encode()).hexdigest())
    return {**value, **changes}


def test_complete_original_header_body_and_commit_are_preserved():
    value = original()
    row = source_row(value, source_sha256='a'*64, position=17)
    assert row['text'] == value['text']
    assert row['id'] == 'static-code:'+'a'*64+':17'
    assert row['source_groups'][0] == 'repo:owner/package'
    assert row['provenance']['commit_sha'] == value['commit_sha']
    assert row['provenance']['code_bytes'] == len('const name = "é";\n'.encode())
    assert 'assistant' not in row and 'execution' not in row


@pytest.mark.parametrize('changes', [
    {'repo':'Other'}, {'text':'const replacement = 1;'}, {'sha256':'0'*64},
    {'size_bytes':999}, {'size_bytes':True}, {'commit_sha':'unknown'},
    {'owner':'Owner\nInjected'}, {'relpath':'src/main.js\nNew header'},
])
def test_changed_or_ambiguous_source_is_pending_not_repaired(changes):
    with pytest.raises(ValueError):
        source_row(original(**changes), source_sha256='a'*64, position=1)


def test_empty_source_is_explicitly_not_a_continuation_target():
    with pytest.raises(ValueError, match='empty original'):
        source_row(original(body=''), source_sha256='a'*64, position=0)


def test_existing_repository_and_equal_content_protection_close_before_conversion():
    a = source_row(original(), source_sha256='a'*64, position=0)
    b = source_row(original(owner='Other', text=original()['text'].replace('Owner/', 'Other/')),
                   source_sha256='a'*64, position=1)
    closed, _ = close_group_sets([a['source_groups'], b['source_groups']],
        {'held': {'repo:owner/package'}, 'quality': set(), 'pending': set()})
    assert disposition(a['source_groups'], closed) == 'test'
    assert disposition(b['source_groups'], closed) == 'test'
    assert disposition(['repo:unrelated/pkg'], closed) == 'train'
    closed['pending'].add('repo:other/package')
    assert disposition(b['source_groups'], closed) == 'pending'


def test_static_graph_keeps_original_vocabulary_and_whole_file():
    from scripts.neuralese_data.build_static_repository_graphs import continuation_graph
    value = original(body='const marker = "<|neuralese_produce|>";\n')
    row = source_row(value, source_sha256='a'*64, position=0)
    class OriginalTokenizer:
        def __call__(self, text, *, add_special_tokens, truncation):
            assert add_special_tokens is False and truncation is False
            return {'input_ids': list(text.encode('utf-8'))}
    class ExpandedTokenizer:
        def __call__(self, *args, **kwargs):
            pytest.fail('new special tokens must not reinterpret original code text')
        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return bytes(ids).decode('utf-8')
    graph = continuation_graph(row, 'train', OriginalTokenizer(), ExpandedTokenizer(), 'a'*64, 'b'*64)
    assert len(graph['nodes']) == 1
    node = graph['nodes'][0]
    assert bytes(node['token_ids']).decode() == value['text']
    assert node['target_weights'] == [0.] + [1.]*(len(node['token_ids'])-1)
    assert not node.get('imports') and not node.get('produces')
    assert graph['conversion']['source_suffix_start'] == 0
    assert graph['conversion']['source_tokenizer_fingerprint'] == 'a'*64


def test_upstream_crlf_size_and_lf_digest_preserve_original_target_bytes():
    body = 'const value = 1;\r\nreturn value;\r\n'
    value = original(body=body, sha256=hashlib.sha256(body.replace('\r\n', '\n').encode()).hexdigest())
    row = source_row(value, source_sha256='a'*64, position=2)
    assert row['text'] == value['text'] and row['text'].endswith('\r\n')
    assert row['provenance']['code_sha256'] == hashlib.sha256(body.encode()).hexdigest()
    assert row['provenance']['upstream_code_sha256'] == value['sha256']
    assert row['provenance']['upstream_digest_law'] == 'original-size-and-crlf-to-lf-digest'
    with pytest.raises(ValueError):
        source_row({**value, 'sha256': '0'*64}, source_sha256='a'*64, position=2)
    with pytest.raises(ValueError):
        source_row({**value, 'size_bytes': len(body.replace('\r\n','\n').encode())}, source_sha256='a'*64, position=2)


def test_tiny_parquet_main_preserves_distinct_newlines_and_durable_accounting(tmp_path, monkeypatch):
    import json
    import sys
    import pyarrow as pa
    import pyarrow.parquet as pq
    import transformers
    from scripts.neuralese_data import build_static_repository_graphs as builder
    from scripts.neuralese_data import build_native_semantic_pairs_e2e as provision_module
    from scripts.neuralese_data import build_verified_teacher_trajectory_graphs as protection_module
    from natlang_neuralese.data import text_corpus
    from natlang_neuralese.common.jsonio import iter_jsonl_rows
    lf = original(body='let value = 1;\n')
    crlf = original(body='let value = 1;\r\n', sha256=lf['sha256'])
    held = original(body='let held = 3;\n', owner='Held',
                    text='<REPO_NAME>Held/Package <FILE_PATH>src/main.js\nlet held = 3;\n')
    quality = original(body='let pending = 4;\n', owner='Quality',
                       text='<REPO_NAME>Quality/Package <FILE_PATH>src/main.js\nlet pending = 4;\n')
    source = tmp_path/'source.parquet'
    pq.write_table(pa.Table.from_pylist([lf, crlf, original(sha256='0'*64), held, quality]), source)
    def binding(path):
        return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    acquisition = tmp_path/'acquisition.json'
    acquisition.write_text(json.dumps(dict(ref='D39', source=binding(source), parquet_rows=5, training_admission=False)))
    protection = tmp_path/'protection.json'; protection.write_text('{}')
    class Tokenizer:
        expanded = False
        def __call__(self, text, *, add_special_tokens, truncation):
            assert not self.expanded and add_special_tokens is False and truncation is False
            return {'input_ids': list(text.encode())}
        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return bytes(ids).decode()
    def provision(tokenizer):
        tokenizer.expanded = True
        return (1000, 1001, 1002, 1003)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    monkeypatch.setattr(text_corpus, 'tokenizer_fingerprint', lambda t: ('b' if t.expanded else 'a')*64)
    monkeypatch.setattr(provision_module, 'provision', provision)
    monkeypatch.setattr(protection_module, 'current_protection', lambda p: dict(
        held={'repo:held/package'}, quality={'repo:quality/package'}, pending=set(), proof=binding(protection)))
    output = tmp_path/'graphs'
    monkeypatch.setattr(sys, 'argv', ['build_static_repository_graphs', '--source',str(source),
        '--source-sha256',binding(source)['sha256'], '--expected-rows','5', '--acquisition',str(acquisition),
        '--acquisition-sha256',binding(acquisition)['sha256'], '--protection-manifest',str(protection),
        '--base-tokenizer',str(tmp_path/'tokenizer'), '--model-key','lfm25', '--output',str(output)])
    builder.main()
    receipt = json.loads((output/'receipt.json').read_text())
    assert receipt['counts'] == dict(train=2, test=1, invalid_source_pending=1, protected_source_pending=1)
    train = [row for _, row in iter_jsonl_rows(output/'train/native-graphs.jsonl.zst')]
    test = [row for _, row in iter_jsonl_rows(output/'test/native-graphs.jsonl.zst')]
    assert len(train)==2 and len(test)==1
    assert [bytes(g['nodes'][0]['token_ids']).decode() for g in train] == [lf['text'], crlf['text']]
    proofs = [g['conversion']['static_repository_source'] for g in train]
    assert proofs[0]['code_sha256'] != proofs[1]['code_sha256']
    assert proofs[0]['upstream_code_sha256'] == proofs[1]['upstream_code_sha256']
    pending = [json.loads(line) for line in (output/'source-pending.jsonl').read_text().splitlines()]
    assert {row['source_row'] for row in pending} == {2,4}
    assert len([json.loads(line) for line in (output/'source-row-proofs.jsonl').read_text().splitlines()]) == 3
    assert receipt['training_admission'] is False and receipt['training_exposure'] == 0
    assert receipt['source']==binding(source) and receipt['acquisition']==binding(acquisition)
    for split in ('train','test'):
        assert receipt['outputs'][split]['sha256']==binding(output/split/'native-graphs.jsonl.zst')['sha256']
    from scripts.neuralese_data.prepare_static_repository_admission import prepare, verify_audit
    from scripts.neuralese_data.partition_acquired_native_capacity import partition
    admission = tmp_path/'admission'
    audit = prepare([output/'receipt.json'], {'lfm25':tmp_path/'tokenizer'}, protection, admission)
    assert audit['by_model']['lfm25'] == receipt['counts']
    assert verify_audit(admission/'audit.json') == audit
    capacity = partition(admission/'audit.json', admission/'source-render-samples.json',
                         {'lfm25':{'forward_tokens':1, 'kv_slots':1}}, tmp_path/'capacity')
    assert capacity['counts']['lfm25']['train:remaining-long']['rows'] == 2
    assert capacity['counts']['lfm25']['test:remaining-long']['rows'] == 1
    assert capacity['counts']['lfm25']['train:fit']['rows'] == 0
    obligations = json.loads((tmp_path/'capacity/files/pending-sources.json').read_text())
    assert len(obligations['rows']) == 2 and all(row['required_visits']==1 for row in obligations['rows'])
    # Even a freshly rebound conversion receipt cannot authorize altered targets.
    from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
    train[0]['nodes'][0]['token_ids'][0] += 1
    writer = GraphWriter(output/'train/native-graphs.jsonl.zst')
    for graph in train:
        writer.write(graph)
    receipt['outputs']['train'] = writer.finish()
    (output/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='evidence binding changed'):
        verify_audit(admission/'audit.json')
    with pytest.raises(ValueError, match='differs from exact replay'):
        prepare([output/'receipt.json'], {'lfm25':tmp_path/'tokenizer'}, protection, tmp_path/'altered-admission')


def test_static_replay_compares_json_types_not_python_numeric_equality():
    from scripts.neuralese_data.prepare_static_repository_admission import next_exact
    with pytest.raises(ValueError, match='differs from exact replay'):
        next_exact(iter([{'token':True}]), {'token':1})
