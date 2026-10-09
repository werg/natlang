import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_machine_paths as c  # noqa: E402


def test_no_new_machine_paths():
    new, _gone = c.compare(c.offenders(), c.load_baseline())
    assert not new, f"new files with absolute machine paths (use common.paths): {new}"


def test_compare_detects_new_and_removed():
    assert c.compare(["a", "b"], ["b", "c"]) == (["a"], ["c"])
