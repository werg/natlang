import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'label_decision_cases', Path(__file__).parents[1] / 'scripts/label_decision_cases.py')
labeler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labeler)


class CheckpointIdentityTests(unittest.TestCase):
    def test_streamed_identity_matches_legacy_for_multiple_and_empty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / 'z.safetensors', root / 'nested' / 'a.json', root / 'empty.json']
            files[1].parent.mkdir()
            files[0].write_bytes(bytes(range(251)) * 5)
            files[1].write_bytes(b'{"checkpoint": [1, 2, 3]}')
            files[2].write_bytes(b'')

            ordered = sorted(map(str, files))
            legacy = hashlib.sha256(b''.join(
                hashlib.sha256(Path(path).read_bytes()).digest() for path in ordered
            )).hexdigest()

            self.assertEqual(labeler.checkpoint_identity_sha256(reversed(ordered), chunk_size=7), legacy)

    def test_chunk_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            labeler.checkpoint_identity_sha256([], chunk_size=0)


if __name__ == '__main__':
    unittest.main()
