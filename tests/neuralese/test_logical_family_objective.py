import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'neuralese_data'))
from build_logical_family_objective import family_of  # noqa: E402


def test_cohort_leaves_map_to_logical_families():
    assert family_of(Path('c/t1-communication-03.jsonl.zst')) == 'native-communication'
    assert family_of(Path('c/t5-broad-00.jsonl.zst')) == 'coding'
    assert family_of(Path('c/t6-broad-01.jsonl.zst')) == 'external-agent'
    assert family_of(Path('c/t2-scope-11.jsonl.zst')) == 'natlang'
    with pytest.raises(ValueError):
        family_of(Path('elsewhere/native-graphs.jsonl.zst'))
