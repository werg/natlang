import hashlib
import json

import pytest

from scripts.prune_reviewed_leaf_references import prune


def test_prune_only_reviewed_sell_rows_and_preserve_backup(tmp_path):
    rows = [
        {"key": "bad", "function": "say", "args": {"action": {"code": "sell_list_price"}}},
        {"key": "good", "function": "say", "args": {"action": {"code": "sell_at_offer"}}},
        {"key": "other", "function": "say", "args": {"action": {"code": "out_of_stock"}}},
        {"key": "new", "function": "page_content", "args": {}},
    ]
    bank = tmp_path / "bank.jsonl"
    bank.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = bank.read_bytes()
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"prefix_lines": 3,
                                  "prefix_sha256": hashlib.sha256(b"".join(source.splitlines(keepends=True)[:3])).hexdigest(),
                                  "expected_scoped_rows": 2, "approved_indices": [1]}))
    backup = tmp_path / "backup.jsonl"

    result = prune(bank, review, backup)
    assert result["quarantined"] == 1
    assert backup.read_bytes() == source
    assert [json.loads(line)["key"] for line in bank.read_text().splitlines()] == ["good", "other", "new"]
    with pytest.raises(FileExistsError):
        prune(bank, review, backup)
