"""Read exact source/packet-bound Host export diagnostics; never admit a trajectory."""
import hashlib
import json
import os


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_rejection_diagnostic(directory, raw, scenario):
    if os.environ.get('VIBESTUDIO_EXPORT_FAILURE_DIAGNOSTICS') == '0':
        return None, []
    path = raw.parent / 'export-rejections.jsonl'
    if not path.is_file():
        return None, []
    errors = []
    matching = None
    raw_hash = _sha(raw)
    pins = directory / 'source.json'
    pins_hash = _sha(pins) if pins.is_file() else None
    for number, line in enumerate(path.read_bytes().splitlines(), 1):
        try:
            row = json.loads(line.decode("utf8"))
            if (not isinstance(row, dict) or row.get('schema') != 'vibestudio.teacher-export-rejection/1'
                    or row.get('scenario') != scenario or row.get('runId') != raw.parent.name
                    or row.get('trajectory') != {'path': raw.name, 'sha256': raw_hash}
                    or pins_hash is None or row.get('pinsSha256') != pins_hash
                    or not isinstance(row.get('error'), dict)
                    or not isinstance(row['error'].get('message'), str) or not row['error']['message']):
                errors.append(f'Unbound export diagnostic at {path.name}:{number}')
                continue
            matching = row
        except (ValueError, TypeError, UnicodeDecodeError):
            errors.append(f'Malformed export diagnostic at {path.name}:{number}')
    return matching, errors
