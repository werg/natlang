"""Exact reviewed decision-ID files shared by text and trajectory assembly."""
import json
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text


def load_decision_holds(paths):
    identities, bindings = set(), []
    for value in paths:
        path = Path(value)
        before = sha256_file_hex(path)
        with open_jsonl_text(path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (not isinstance(row.get('id'), str) or not row['id'] or
                        not isinstance(row.get('reason'), str) or not row['reason']):
                    raise ValueError(f'invalid native decision hold in {path}')
                identities.add(row['id'])
        if sha256_file_hex(path) != before:
            raise ValueError(f'native decision holds changed during snapshot: {path}')
        bindings.append({'kind': 'decision-hold', 'path': str(path), 'sha256': before,
                         'bytes': path.stat().st_size})
    return identities, bindings
