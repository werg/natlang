"""Lone UTF-16 surrogates (half an emoji from JS code-unit slicing, decoded by json.loads from a `\\ud83d` escape)
never break the UTF-8 writers of trajectory, episode and teacher text (common/jsonio SURROGATE_ERRORS)."""
import json
import sys
from pathlib import Path

import pytest
import zstandard

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))
sys.path.insert(0, str(ROOT / 'training/neuralese'))

from natlang_neuralese.common import jsonio  # noqa: E402

# What the vibestudio 0058 capture held: a test error with the high half of U+1F642 alone.
ROW = json.loads('{"error": "expected \'\\ud83d\' to deeply equal \'\\ud83d\\ude42\'", "\\udc00key": 1}')


def test_scrub_replaces_lone_and_joins_split_pairs():
    assert jsonio.scrub_surrogates('a\ud83db') == 'a\ufffdb'
    assert jsonio.scrub_surrogates('\ude42') == '\ufffd'
    assert jsonio.scrub_surrogates('\ud83d\ude42') == '\U0001f642'   # a split pair is the character it encodes
    plain = 'ok \U0001f642 é'
    assert jsonio.scrub_surrogates(plain) is plain
    assert jsonio.utf8('x\ud83d') == 'x\ufffd'.encode()


def test_canonical_json_unchanged_without_surrogates_and_encodable_with_them():
    value = {'b': 'é\U0001f642', 'a': [1, 2]}
    assert jsonio.canonical_json_str(value) == json.dumps(value, ensure_ascii=False, sort_keys=True,
                                                         separators=(',', ':'))
    assert json.loads(jsonio.canonical_json_bytes(ROW)) == {'error': "expected '\ufffd' to deeply equal '\U0001f642'",
                                                             '\ufffdkey': 1}


def test_jsonl_text_writers(tmp_path):
    with jsonio.write_jsonl_text(tmp_path / 'a.jsonl') as stream:
        stream.write(json.dumps(ROW, ensure_ascii=False) + '\n')
    assert '\ufffd' in (tmp_path / 'a.jsonl').read_text()
    with jsonio.write_jsonl_text(tmp_path / 'a.jsonl.zst') as stream:
        stream.write(json.dumps(ROW, ensure_ascii=False) + '\n')
    assert '\ufffd' in zstandard.ZstdDecompressor().decompress((tmp_path / 'a.jsonl.zst').read_bytes(),
                                                               max_output_size=1 << 20).decode()
    assert jsonio.write_canonical_jsonl(tmp_path / 'c.jsonl', [ROW]) == 1


def test_zstd_writer_exports_lone_surrogates(tmp_path):
    import sft_sources
    writer = sft_sources.ZstdWriter(tmp_path / 'x.jsonl.zst')
    writer.write(ROW)
    writer.close()
    rows = [json.loads(line) for line in sft_sources.zstd_lines(tmp_path / 'x.jsonl.zst')]
    assert rows[0]['error'] == "expected '\ufffd' to deeply equal '\U0001f642'"


def test_recorded_turns_and_port_records(tmp_path):
    import common
    import recorded_turns
    recorded_turns.write(tmp_path, 'episode', [{'messages': [{'role': 'tool', 'content': ROW['error']}]}])
    assert '\ufffd' in zstandard.ZstdDecompressor().decompress(recorded_turns.path(tmp_path, 'episode').read_bytes(),
                                                               max_output_size=1 << 20).decode()
    sink = common.Sink(tmp_path, 'demo')
    sink.accept({'split': 'train', 'text': ROW['error']})
    sink.reject('store', 0, 'bad: ' + ROW['error'])
    sink.close()
    assert '\ufffd' in (tmp_path / 'demo.port-records.jsonl').read_text()


def test_opt_out_keeps_the_strict_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_KEEP_LONE_SURROGATES', '1')
    with pytest.raises(UnicodeEncodeError):
        jsonio.canonical_json_bytes(ROW)
    with pytest.raises(UnicodeEncodeError):
        jsonio.utf8('\ud83d')
    with pytest.raises(UnicodeEncodeError):
        with jsonio.write_jsonl_text(tmp_path / 'b.jsonl') as stream:
            stream.write('\ud83d\n')


def test_teacher_receipt_canonical_bytes_match_previous_encoding():
    from natlang_neuralese.serve.teacher_source_receipt import _canonical
    value = {'z': 'é', 'a': ['\U0001f642']}
    assert _canonical(value) == json.dumps(value, ensure_ascii=False, sort_keys=True,
                                           separators=(',', ':')).encode()
    assert b'\xef\xbf\xbd' in _canonical({'t': '\ud83d'})
