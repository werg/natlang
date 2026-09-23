import json

import pytest

from scripts.audit_training_corpus import assess, audit_corpus, report_for
from scripts.render_training_corpus import _file_sha, _tokenizer_info


class Tokenizer:
    chat_template = 'test-template'
    eos_token = '!'
    name_or_path = 'mock'

    def __init__(self):
        self.calls = 0

    def get_vocab(self):
        return {'a': 1, '!': 2}

    def __call__(self, value, *, add_special_tokens):
        assert not add_special_tokens
        self.calls += 1
        return {'input_ids': list(value.encode())}


def row(id='one', **extra):
    return {'id': id, 'prompt': 'hi', 'completion': 'ok!', 'split': 'train',
            'family': 'arrays', 'source': {'name': 'repo-one'},
            'training_admission': {'approved': True}, **extra}


def source_file(tmp_path, rows, tokenizer=None):
    path = tmp_path / 'rendered.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    _, renderer = _tokenizer_info(tokenizer or Tokenizer(), 'mock', None)
    path.with_name(path.name + '.manifest.json').write_text(json.dumps({
        'renderer': renderer, 'rows': len(rows), 'sha256': _file_sha(path)}))
    return path


def test_audit_counts_tokens_and_filters_without_truncation(tmp_path):
    source = source_file(tmp_path, [row(), row('large', completion='x' * 30 + '!'),
                                   row('held', split='test', completion='y!')])
    out = tmp_path / 'ready.jsonl'
    assert audit_corpus(source, out, model='mock', tokenizer=Tokenizer(), max_len=10) == 0
    accepted = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r['id'] for r in accepted] == ['one', 'held']
    assert accepted[0]['completion'] == 'ok!'
    assert accepted[0]['token_counts'] == {'prompt_tokens': 2, 'supervised_tokens': 3, 'total_tokens': 5}
    report = json.loads(out.with_name(out.name + '.audit.json').read_text())
    assert report['rejections'] == {'over_token_budget': 1}
    assert report['tokens']['supervised_tokens'] == 5
    assert report['training_tokens']['supervised_tokens'] == 3
    assert report['distributions']['split']['train']['usable_rows'] == 1
    assert report['distributions']['split']['test']['supervised_tokens'] == 2
    assert report['distributions']['source']['repo-one']['supervised_token_fraction'] == 1


@pytest.mark.parametrize('change,reason', [
    ({'prompt': ''}, 'empty_or_invalid_pair'),
    ({'completion': '!'}, 'empty_assistant_target'),
    ({'completion': 'no stop'}, 'invalid_target_termination'),
    ({'completion': 'mid!tail!'}, 'invalid_target_termination'),
    ({'training_admission': {}}, 'not_explicitly_admitted'),
    ({'split': None}, 'missing_or_invalid_split'),
])
def test_admission_reasons(change, reason):
    result = assess(row(**change), Tokenizer(), '!', 100)
    assert result['record'] is None
    assert result['summary']['reason'] == reason


def test_token_boundary_mismatch_is_not_silently_trained():
    class MergingTokenizer(Tokenizer):
        def __call__(self, value, **kwargs):
            result = super().__call__(value, **kwargs)
            if value == 'hiok!':
                result['input_ids'] = [123]
            return result
    assert assess(row(), MergingTokenizer(), '!', 100)['summary']['reason'] == 'tokenization_boundary_mismatch'


def test_stop_resume_reuses_chunks_and_detects_corruption(tmp_path):
    tok = Tokenizer()
    source = source_file(tmp_path, [row(str(i)) for i in range(3)], tok)
    out = tmp_path / 'out.jsonl'
    assert audit_corpus(source, out, model='mock', tokenizer=tok, chunk_rows=1, should_stop=lambda: True) == 75
    assert tok.calls == 3
    assert not out.exists()
    assert audit_corpus(source, out, model='mock', tokenizer=tok, chunk_rows=1) == 0
    assert tok.calls == 9
    assert audit_corpus(source, out, model='mock', tokenizer=tok, chunk_rows=1) == 0
    assert tok.calls == 9
    chunk = out.with_name(out.name + '.audit-cache') / 'chunk-00000000.jsonl'
    chunk.write_text('corrupt')
    with pytest.raises(ValueError, match='cache corruption'):
        audit_corpus(source, out, model='mock', tokenizer=tok, chunk_rows=1)


def test_context_change_and_output_tampering_reject_resume(tmp_path):
    source = source_file(tmp_path, [row()])
    out = tmp_path / 'out.jsonl'
    audit_corpus(source, out, model='mock', tokenizer=Tokenizer(), max_len=10)
    with pytest.raises(ValueError, match='identity changed'):
        audit_corpus(source, out, model='mock', tokenizer=Tokenizer(), max_len=20)
    out.write_text('tampered')
    with pytest.raises(ValueError, match='output corruption'):
        audit_corpus(source, out, model='mock', tokenizer=Tokenizer(), max_len=10)


def test_renderer_identity_and_source_hash_are_checked(tmp_path):
    source = source_file(tmp_path, [row()])
    with pytest.raises(ValueError, match='mismatch: model'):
        audit_corpus(source, tmp_path / 'out', model='other', tokenizer=Tokenizer())
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='corpus hash mismatch'):
        audit_corpus(source, tmp_path / 'out', model='mock', tokenizer=Tokenizer())


def test_no_usable_train_rows_blocks_gate_but_writes_report(tmp_path):
    source = source_file(tmp_path, [row(split='test')])
    out = tmp_path / 'out'
    assert audit_corpus(source, out, model='mock', tokenizer=Tokenizer()) == 2
    assert json.loads(out.with_name(out.name + '.audit.json').read_text())['ready'] is False


def test_duplicate_and_evidence_counts_do_not_claim_spec_correctness():
    rows = [row(implementation_sha256='same', execution_verified=True),
            row('two', implementation_sha256='same', behavioral_evidence={'kind': 'upstream_assertion'})]
    report = report_for([assess(r, Tokenizer(), '!', 100) for r in rows], 100)
    assert report['duplicate_clusters']['implementation']['excess_rows'] == 1
    assert report['duplicate_clusters']['pair_sha256']['excess_rows'] == 1
    assert set(report['distributions']['evidence']) == {'execution_verified_unspecified_oracle', 'upstream_assertion'}


def test_manifest_missing_after_commit_is_recovered_idempotently(tmp_path):
    source = source_file(tmp_path, [row()])
    out = tmp_path / 'out'
    audit_corpus(source, out, model='mock', tokenizer=Tokenizer())
    original = out.read_bytes()
    manifest = out.with_name(out.name + '.manifest.json')
    manifest.unlink()
    assert audit_corpus(source, out, model='mock', tokenizer=Tokenizer()) == 0
    assert manifest.exists() and out.read_bytes() == original


def test_upstream_rejections_are_reported_separately_and_pinned(tmp_path):
    source = source_file(tmp_path, [row()])
    out = tmp_path / 'out'
    ledger = tmp_path / 'rejections.jsonl'
    ledger.write_text(json.dumps({'id': 'missing', 'reason': 'missing behavioral description'}) + '\n')
    assert audit_corpus(source, out, model='mock', tokenizer=Tokenizer(), rejection_ledgers=[ledger]) == 0
    report = json.loads(out.with_name(out.name + '.audit.json').read_text())
    assert report['rejected_rows'] == 0
    assert report['upstream_rejection_ledgers'][str(ledger)]['reasons'] == {'missing behavioral description': 1}
    ledger.write_text('')
    with pytest.raises(ValueError, match='identity changed'):
        audit_corpus(source, out, model='mock', tokenizer=Tokenizer(), rejection_ledgers=[ledger])
