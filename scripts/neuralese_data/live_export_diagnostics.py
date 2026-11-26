"""Read exact source/packet-bound Host export diagnostics; never admit a trajectory."""
import hashlib
import json
import os
import re


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
            review = row.get('reviewFailure')
            if review is not None:
                capture = review.get('capture') if isinstance(review, dict) else None
                relative = capture.get('path') if isinstance(capture, dict) else None
                target = directory / relative if isinstance(relative, str) else None
                if (target is None or not target.resolve().is_relative_to(directory.resolve())
                        or not target.is_file() or _sha(target) != capture.get('sha256')):
                    errors.append(f'Unbound review failure capture at {path.name}:{number}')
                    continue
                actual = json.loads(target.read_text())
                pins_value = json.loads(pins.read_text())
                failure = actual.get('failure') or {}
                error = failure.get('error') or {}
                transport = (error.get('code') in ('ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT', 'EAI_AGAIN', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_SOCKET')
                             or bool(re.match(r'^(?:fetch failed$|socket hang up$|Client network socket disconnected before secure TLS connection (?:was )?established|(?:connect |read )?(?:ECONNRESET|ECONNREFUSED|ETIMEDOUT|EAI_AGAIN)\b|getaddrinfo (?:EAI_AGAIN|ETIMEDOUT)\b|upstream HTTP (?:408|429|5[0-9]{2})\b)', error.get('message', ''))))
                if (actual.get('schema') != 'natlang.live-workflow-review-failure/1'
                        or actual.get('complete') is not False or actual.get('verdict') is not None
                        or not isinstance(actual.get('teacher'), str) or actual.get('teacher') != pins_value.get('teacher')
                        or not isinstance(actual.get('model'), str) or actual['model'] == actual['teacher']
                        or not isinstance(actual.get('input'), dict)
                        or review.get('schema') != actual['schema'] or review.get('model') != actual['model']
                        or review.get('teacher') != actual['teacher'] or review.get('failure') != failure):
                    errors.append(f'Invalid review failure binding at {path.name}:{number}')
                    continue
                row['review_transport_failure'] = (transport and failure.get('kind') == 'provider-transport'
                                                   and failure.get('phase') in ('provider-prepare', 'provider-turn'))
            matching = row
        except (ValueError, TypeError, UnicodeDecodeError):
            errors.append(f'Malformed export diagnostic at {path.name}:{number}')
    return matching, errors
