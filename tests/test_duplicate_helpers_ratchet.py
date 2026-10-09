import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_duplicate_helpers as c  # noqa: E402


def test_duplicate_helpers_only_fall():
    grown, _shrunk = c.compare(c.counts(), c.load_baseline())
    assert not grown, f"duplicate helper definitions grew: {grown}"


def test_compare_semantics():
    assert c.compare({"a": 2, "n": 1}, {"a": 1, "b": 3}) == (["a: 1 -> 2", "n: 0 -> 1"], ["b: 3 -> 0"])
