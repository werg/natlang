"""Physical storage bindings preserve the decoded corpus identity."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('register_render', Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/register_native_turn_render.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_plain_and_compressed_bindings_name_the_actual_files():
    plain = {'text_data_sha256': 'a' * 64}
    assert module.text_binding('full', 'mellum21', plain) == {'file': 'full/mellum21/text-data.jsonl', 'sha256': 'a' * 64}
    compressed = {**plain, 'text_data_file': 'text-data.jsonl.zst',
                  'decoded_text_data_sha256': 'b' * 64, 'decoded_text_data_bytes': 12345}
    assert module.text_binding('full', 'mellum21', compressed) == {
        'file': 'full/mellum21/text-data.jsonl.zst', 'sha256': 'a' * 64,
        'compression': 'zstd', 'decoded_sha256': 'b' * 64, 'decoded_bytes': 12345}


@pytest.mark.parametrize('extra', [{}, {'decoded_text_data_sha256': 'bad', 'decoded_text_data_bytes': 1},
                                  {'decoded_text_data_sha256': 'b' * 64, 'decoded_text_data_bytes': True}])
def test_compressed_binding_requires_a_complete_decode_proof(extra):
    with pytest.raises(ValueError, match='verified decoded'):
        module.text_binding('full', 'lfm25', {'text_data_file': 'text-data.jsonl.zst', 'text_data_sha256': 'a' * 64, **extra})


def test_storage_filename_cannot_escape_the_render_directory():
    with pytest.raises(ValueError, match='filename'):
        module.text_binding('full', 'lfm25', {'text_data_file': '../../../other.jsonl', 'text_data_sha256': 'a' * 64})
