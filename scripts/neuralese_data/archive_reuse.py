"""Readonly content comparison for already archived seed episodes."""
import json
import os
from pathlib import Path
import sys

# The same small helper runs over SSH stdin. Collection stays on its own host;
# shared hashing comes from the existing archive reader's checkout.
if __name__ == '__main__':
    repo = Path(sys.argv[1])
    sys.path[:0] = [str(repo / 'training/neuralese'), str(repo / 'scripts/neuralese_data')]
    # Each SSH helper is short-lived; avoid indexing unrelated model/corpus
    # identities from the archive host's global cache on every seed check.
    os.environ.setdefault('NATLANG_HASH_CACHE', str(Path.home() / '.cache/natlang/vibestudio-seed-sha256.jsonl'))

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from retained_state_archive import inventory


def capture_signature(root, exclusions=()):
    root = Path(root)
    omitted = [root.name + '/' + name.strip('/') for name in exclusions]
    before = inventory(root, hashes=False, omitted=omitted)
    content = []
    for entry in before:
        relative = Path(entry['path']).relative_to(root.name)
        row = {key: entry[key] for key in ('kind', 'mode', 'bytes', 'link') if key in entry}
        row['path'] = relative.as_posix()
        if entry['kind'] == 'file':
            row['sha256'] = sha256_file_hex_cached(root / relative, trust_archive_receipt=False)
        content.append(row)
    if inventory(root, hashes=False, omitted=omitted) != before:
        raise ValueError(f'capture changed while checking archive reuse: {root}')
    return canonical_json_sha256_hex(content)


def archive_matches(destination, records, signature, exclusions=()):
    destination = Path(destination)
    missing = False
    for name, expected in records.items():
        path = destination / name
        if not path.exists():
            missing = True
        elif json.loads(path.read_text()) != expected:
            raise ValueError(f'immutable archive record differs: {path}')
    capture = destination / 'capture'
    if missing or not capture.is_dir() or capture.is_symlink():
        return False
    return capture_signature(capture, exclusions) == signature


if __name__ == '__main__':
    request = json.loads(sys.argv[2])
    print(json.dumps(archive_matches(**request)))
