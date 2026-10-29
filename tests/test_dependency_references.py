import subprocess

import pytest

from scripts.coreweave.dependency_references import reference_paths


def test_batch_reference_scan_matches_independent_searches_across_windows(tmp_path):
    first, second, absent = 'a' * 64, 'b' * 64, 'c' * 64
    one = tmp_path / 'one: file.json'
    one.write_text('x' * 63 + first + 'x' * 70 + second)
    two = tmp_path / 'two\nlines.json'
    two.write_text(second)
    (tmp_path / 'ignored.txt').write_text(absent)
    actual = reference_paths([first, second, absent], [tmp_path], chunk_bytes=64)
    for value in (first, second, absent):
        check = subprocess.run(['rg', '-l', '--null', '-F', value, '--glob', '*.json', str(tmp_path)],
                               capture_output=True, check=False)
        expected = {p.decode() for p in check.stdout.split(b'\0') if p}
        assert actual[value] == expected
    assert actual[first] == {str(one)} and actual[second] == {str(one), str(two)}


def test_scan_fails_closed_on_search_failure_and_invalid_identity(tmp_path):
    with pytest.raises(RuntimeError, match='search failed'):
        reference_paths(['a' * 64], [tmp_path / 'missing'])
    with pytest.raises(ValueError, match='SHA-256'):
        reference_paths(['not-a-hash'], [tmp_path])
    assert reference_paths([], [tmp_path]) == {}
