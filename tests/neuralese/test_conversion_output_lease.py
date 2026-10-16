from pathlib import Path

import pytest

from scripts.neuralese_data.conversion_output_lease import acquire_output_lease


def test_output_lease_excludes_a_second_writer_and_releases_on_close(tmp_path):
    output = Path(tmp_path) / 'render'
    with acquire_output_lease(output) as first:
        with pytest.raises(RuntimeError, match='already owns'):
            acquire_output_lease(output)
        first.seek(0)
        first_record = first.read()
        assert 'pid' in first_record
    with acquire_output_lease(output) as second:
        assert second.fileno() >= 0
