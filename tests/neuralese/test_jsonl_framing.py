"""JSONL framing preserves valid Unicode and exact object-row diagnostics."""
import json
import pytest
from natlang_neuralese.common.jsonio import read_jsonl,write_canonical_jsonl
from natlang_neuralese.eval import task_packet

def test_shared_task_and_probe_reader_preserves_unicode_separators(tmp_path):
    rows=[{'id':str(i),'messages':[{'role':'user','content':'A\u0085B\u2028C\u2029D💡'}]}for i in range(48)]
    path=tmp_path/'probes.jsonl';write_canonical_jsonl(path,rows)
    # This fixture reproduces the old string-split parser's row fragmentation.
    assert len(path.read_text().splitlines())>48
    assert read_jsonl(path)==rows
    assert task_packet.read_jsonl is read_jsonl
    assert task_packet.read_jsonl(path)==rows

def test_jsonl_object_errors_keep_physical_line_number(tmp_path):
    path=tmp_path/'rows.jsonl';path.write_text('\n[1,2]\n')
    with pytest.raises(ValueError,match=r'expected an object at .*:2'):
        read_jsonl(path)

def test_jsonl_crlf_and_final_row_without_newline(tmp_path):
    rows=[{'text':'A\u2028B'},{'text':'last\u2029'}];path=tmp_path/'rows.jsonl'
    path.write_bytes((json.dumps(rows[0],ensure_ascii=False)+'\r\n\r\n'+json.dumps(rows[1],ensure_ascii=False)).encode())
    assert read_jsonl(path)==rows
