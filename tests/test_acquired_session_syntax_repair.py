import copy
import hashlib
import json

import pytest

from scripts.neuralese_data.acquired_session_syntax_repair import (
    REPAIR_CLASS,
    REPAIR_RULE,
    argument_repair_record,
    build_repair_view,
    repair_unquoted_query_argument,
    verify_derived_repair_export,
)
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable, write_jsonl_text


SCHEMA = {
    'type': 'object',
    'properties': {
        'query': {'type': 'string'},
        'topn': {'type': 'number', 'default': 10},
        'source': {'type': 'string'},
    },
    'required': ['query'],
}


@pytest.mark.parametrize('original,expected', [
    ('{"query": Global Economic Prospects 2025 pdf, "topn": 10}',
     'Global Economic Prospects 2025 pdf'),
    (r'{"query": \"SMBC\" \"Tokyo Disneyland\" \"branch\"", "topn": 10, "source": "news"}',
     r'\"SMBC\" \"Tokyo Disneyland\" \"branch\""'),
    (r'{ "query" : \"Sochi 2014 medal table pdf\" , "topn": 10, "source": "news" }',
     r'\"Sochi 2014 medal table pdf\"'),
])
def test_repair_adds_json_string_encoding_without_rewriting_query_characters(original, expected):
    repaired = repair_unquoted_query_argument(original, schema=SCHEMA)
    assert repaired is not None
    decoded = json.loads(repaired)
    assert decoded['query'] == expected
    if '"source"' in original:
        assert decoded['topn'] == 10 and decoded['source'] == 'news'
    # The original is immutable; only the argument string is transformed.
    saved = copy.deepcopy(original)
    assert original == saved


@pytest.mark.parametrize('original', [
    # Duplicate delimiter makes the boundary ambiguous.
    '{"query": one, "topn": 10, "source": "news", "query": two, "topn": 10}',
    # Damaged tail/schema key.
    '{"query": Carl Perkins first album, Sun":"source": "news", "topn": 10}',
    # Truncated query has no complete delimiter or object.
    '{"query": "President Obama" "Winston\n\n',
    # Control characters are not normalized into JSON escapes.
    '{"query": first\nsecond, "topn": 10}',
    # Unrecognized argument schema and field-name variants fail closed.
    '{"query": a complete query, "top_n": 10, "source": "news"}',
])
def test_ambiguous_truncated_or_nonmatching_source_stays_pending(original):
    assert repair_unquoted_query_argument(original, schema=SCHEMA) is None


def test_schema_must_be_the_declared_browser_search_contract():
    assert repair_unquoted_query_argument('{"query": text, "topn": 10}', schema={
        'type': 'object', 'properties': {'query': {'type': 'string'}, 'limit': {'type': 'number'}},
        'required': ['query']}) is None


def test_repair_record_binds_exact_source_argument_and_classification():
    original = '{"query": exact words here, "topn": 10}'
    repaired = repair_unquoted_query_argument(original, schema=SCHEMA)
    record = argument_repair_record(source={'path': 'source.jsonl.zst', 'sha256': 'a' * 64},
        source_row='part.parquet:17', session_id='session:17', message_index=3,
        call_index=0, tool_name='browser.search', schema=SCHEMA,
        original=original, repaired=repaired)
    assert record['repair_class'] == REPAIR_CLASS
    assert record['repair_rule'] == REPAIR_RULE
    assert record['original_arguments'] == original
    assert json.loads(record['repaired_arguments'])['query'] == 'exact words here'
    assert record['training_admission'] is False
    assert record['tool_execution_verified'] is False
    with pytest.raises(ValueError, match='exact approved syntax transform'):
        argument_repair_record(source={'path': 'source.jsonl.zst', 'sha256': 'a' * 64},
            source_row='part.parquet:17', session_id='session:17', message_index=3,
            call_index=0, tool_name='browser.search', schema=SCHEMA,
            original=original, repaired='{"query":"guessed"}')


def test_derived_view_is_bound_to_exact_parent_and_only_repairs_selected_row(tmp_path):
    parent_dir = tmp_path / 'parent'
    parent_dir.mkdir()
    source_path = parent_dir / 'full-trajectories.jsonl.zst'
    source_file = {'path': 'seed_47/train-00001-of-00003.parquet', 'bytes': 100,
                   'sha256': 'a' * 64}
    args = '{"query": exact literal query, "topn": 10}'
    row = {'trajectory': {'id': 'source-session-1', 'tools': [
                {'function': {'name': 'browser.search', 'parameters': SCHEMA}}],
            'messages': [{'role': 'assistant', 'tool_calls': [
                {'function': {'name': 'browser.search', 'arguments': args}}]}],
            'outcome': {'label': 'success'}, 'split_groups': ['train']},
        'source_row': 'train.parquet:1', 'source_file': source_file,
        'source_revision': 'rev-1', 'positive_candidate': True}
    with write_jsonl_text(source_path) as stream:
        stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    source_sha = sha256_file_hex(source_path)
    parent_export = {'schema': 'natlang.normalized-agent-source-file/1',
        'complete_sessions': 1, 'positive_candidates': 1,
        'contract': {'source': source_file, 'source_rows': 2034,
            'revision': 'rev-1', 'exporter_sha256': 'b' * 64,
            'trajectory_representation': 'existing complete Trajectory dataclass'},
        'outputs': {source_path.name: source_sha}, 'training_admission': False}
    parent_export_path = parent_dir / 'export.json'
    write_json_durable(parent_export_path, parent_export)
    row_sha = canonical_json_sha256_hex(row)
    candidate = {'schema': 'natlang.source16-tool-repair-candidate-extract/1',
        'source': {'path': str(source_path), 'sha256': source_sha}, 'rows': [{
            'id': 'source-session-1', 'source_row': row['source_row'],
            'source_file': source_file, 'normalized_source_sha256': row_sha,
            'calls': [{'message_index': 0, 'call_index': 0, 'name': 'browser.search',
                'schema': SCHEMA, 'arguments': args,
                'arguments_utf8_sha256': hashlib.sha256(args.encode()).hexdigest()}]}]}
    candidate_path = tmp_path / 'candidate.json'
    write_json_durable(candidate_path, candidate)
    audit = {'schema': 'natlang.source-tool-pending-audit/1',
        'source': candidate['source'], 'rows': [{'id': 'source-session-1',
            'failures': {'lfm2': 'tool-arguments: invalid JSON', 'mellum': 'tool-arguments: invalid JSON'}, 'calls': [
            {'message_index': 0, 'call_index': 0, 'decoded_type': 'invalid-json'}]}]}
    audit_path = tmp_path / 'audit.json'
    write_json_durable(audit_path, audit)
    out = tmp_path / 'derived'
    result = build_repair_view(parent_export_path=parent_export_path,
        candidate_audit_path=candidate_path, pending_audit_path=audit_path, output_dir=out)
    assert result['selected_sessions'] == 1
    assert result['repair_calls'] == 1
    derived_export = json.loads((out / 'export.json').read_text())
    assert derived_export['contract']['source_rows'] == 2034
    derived_source = out / 'full-trajectories.jsonl.zst'
    proof = verify_derived_repair_export(out / 'export.json', derived_export, derived_source)
    assert proof['selected_rows'] == 1 and proof['repair_calls'] == 1
    assert derived_export['training_admission'] is False
    assert (out / 'complete.json').exists()
