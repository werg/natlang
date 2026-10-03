import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_lfm_test_runtime_packet import prepare, sha256


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


class LfmTestRuntimePacketTests(unittest.TestCase):
    def test_builds_test_only_packet_and_separate_gold_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "joint.jsonl"
            ir = {"id": "case1", "source_groups": ["held-group"],
                  "semantics": {"expected": True}}
            rows = [
                {"id": "trainrow", "split": "train", "program_id": "train-program",
                 "source_groups": ["train-group"], "source_ids": ["train-source"]},
                {"id": "testrow1", "split": "test", "program_id": "test-program",
                 "source_groups": ["held-group"], "source_ids": ["held-source"],
                 "task": {"program_ir": ir}},
                {"id": "testrow2", "split": "test", "program_id": "test-program",
                 "source_groups": ["held-group"], "source_ids": ["held-source"],
                 "task": {"program_ir": ir}},
            ]
            write_rows(source, rows)
            source_manifest = root / "joint.manifest.json"
            source_manifest.write_text(json.dumps({"version": "test/1", "sha256": sha256(source)}))
            out = root / "packet"
            receipt = prepare(source, out, source_manifest=source_manifest)
            self.assertEqual(receipt["protected_split"]["test_programs"], 1)
            case = json.loads((out / "cases.ir.jsonl").read_text())
            gold = json.loads((out / "gold-reference.jsonl").read_text())
            selection = json.loads((out / "selection.jsonl").read_text())
            self.assertEqual(case["id"], gold["id"])
            self.assertEqual(gold["expected"], True)
            self.assertEqual(selection["split"], "test")

    def test_rejects_source_alias_crossing_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "joint.jsonl"
            ir = {"id": "case1", "semantics": {"expected": 1}}
            write_rows(source, [
                {"id": "t", "split": "train", "program_id": "p1", "source_ids": ["shared"]},
                {"id": "e", "split": "test", "program_id": "p2", "source_ids": ["shared"],
                 "task": {"program_ir": ir}},
            ])
            manifest = root / "joint.manifest.json"
            manifest.write_text(json.dumps({"sha256": sha256(source)}))
            with self.assertRaisesRegex(ValueError, "aliases cross"):
                prepare(source, root / "packet", source_manifest=manifest)


if __name__ == "__main__":
    unittest.main()
