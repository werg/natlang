import hashlib
import tracemalloc

import pytest

from natlang_neuralese.common.jsonio import canonical_json_bytes
from natlang_neuralese.common.ordered_json_sequence import canonical_json_sequence_sha256_hex


def _legacy_digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@pytest.mark.parametrize('values', [
    [],
    (),
    [{'z': 'naïve/雪', 'a': [True, None, -3, 1.25]}, {'tuple': (1, 'λ')}],
    [{'floats': [0.0, -0.0, 1e-300, 1e+300], 'nested': {'b': 2, 'a': 1}}],
])
def test_sequence_digest_is_exact_canonical_array_digest(values):
    assert canonical_json_sequence_sha256_hex(values) == _legacy_digest(values)


def test_sequence_digest_rejects_non_sequence_values():
    with pytest.raises(TypeError, match='list or tuple'):
        canonical_json_sequence_sha256_hex({'row': 1})


def test_large_sequence_digest_keeps_temporary_memory_bounded():
    values = [{'id': i, 'binding': f'{i:064x}', 'groups': [f'group-{i % 101}', 'shared'],
               'forward_tokens': i % 4096, 'supervised_tokens': i % 256}
              for i in range(100_000)]

    tracemalloc.start()
    expected = _legacy_digest(values)
    _, legacy_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    actual = canonical_json_sequence_sha256_hex(values)
    _, streaming_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert actual == expected
    assert streaming_peak < legacy_peak // 4
