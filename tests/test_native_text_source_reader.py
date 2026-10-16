"""The native text converter uses the same symlink-safe reader as the trainer."""
import json
import subprocess

import pytest

from scripts.neuralese_data.build_native_graph_text_corpus import read_jsonl_zstd,numerical_runtime


def test_source_file_symlink_keeps_all_decoded_rows(tmp_path):
    rows=[{'id':'first','content':'literal\u2028line'}, {'id':'second','split':'train'}]
    source=tmp_path/'original.jsonl.zst'
    source.write_bytes(subprocess.run(['zstd','-qc'],input=''.join(json.dumps(row)+'\n' for row in rows).encode(),
                                     capture_output=True,check=True).stdout)
    alias=tmp_path/'source link.jsonl.zst';alias.symlink_to(source)
    assert list(read_jsonl_zstd(alias))==rows


def test_corrupt_compressed_source_cannot_become_empty_success(tmp_path):
    source=tmp_path/'broken.jsonl.zst';source.write_bytes(b'not a compressed frame')
    with pytest.raises(RuntimeError,match='zstd failed'):
        list(read_jsonl_zstd(source))


def test_conversion_numerical_settings_are_observations(monkeypatch):
    import torch
    monkeypatch.setenv('OMP_NUM_THREADS','123')
    result=numerical_runtime()
    assert result['torch_threads']==torch.get_num_threads()
    assert result['thread_environment']['OMP_NUM_THREADS']=='123'
    assert result['torch_version']==str(torch.__version__)
