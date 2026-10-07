import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/assemble_admitted_neuralese_cohort.py"
SPEC = importlib.util.spec_from_file_location("assemble_admitted_neuralese_cohort", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class AssemblerInvariantTests(unittest.TestCase):
    def test_append_preserves_prefix_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "prefix.jsonl"
            output = root / "out.jsonl"
            prefix.write_bytes(b'{"id":"old"}\r\n')
            builder.write_append(prefix, output, [{"id": "new", "split": "train"}])
            self.assertTrue(output.read_bytes().startswith(prefix.read_bytes()))
            self.assertEqual(len(list(builder.rows(output))), 2)

    def test_piece_conflict_is_rejected_without_overwriting_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, delta, output = (root / x for x in ("base", "delta", "out"))
            base.write_text('{"name":"p","text":"base"}\n')
            delta.write_text('{"name":"p","text":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "piece name conflict"):
                builder.merge_pieces(base, delta, output)
            self.assertFalse(output.exists())

    def test_group_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in [
                {"id": "a", "split": "train", "source_groups": ["g"], "source_ids": ["s1"]},
                {"id": "b", "split": "test", "source_groups": ["g"], "source_ids": ["s2"]},
            ]))
            with self.assertRaisesRegex(ValueError, "source groups cross splits"):
                builder.audit_splits(path)


if __name__ == "__main__":
    unittest.main()
