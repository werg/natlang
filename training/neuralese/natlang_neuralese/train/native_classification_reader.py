"""Read original classification proofs through a byte-identical relocated producer.

The classifier's own source remains unchanged: its historical producer identity
is a publication dependency. This adapter creates a separately recorded reader
view; source, rows, indexes and parser-policy bindings are never substituted.
"""
import copy
import json
from pathlib import Path

from ..common.jsonio import write_bytes_durable
from . import native_source_classification as classifier


def migration_classes(proof_path, source, *, directory):
    original = classifier._binding(proof_path)
    value = json.loads(Path(proof_path).read_bytes())
    current = classifier._binding(classifier.__file__)
    relocations = []
    checked = {}
    for name in ('indexes', 'held_indexes'):
        for index, part in enumerate(value.get(name, [])):
            previous = part.get('producer') or {}
            key = (previous.get('path'), previous.get('sha256'), previous.get('bytes'))
            if key not in checked:
                classifier._checked(previous)
                checked[key] = classifier._binding(previous['path'])
            actual = checked[key]
            if any(actual[key] != current[key] for key in ('sha256', 'bytes')):
                raise ValueError('classification reader requires byte-identical historical producer')
            if previous != current:
                relocations.append((name, index, dict(previous)))
    if not relocations:
        return classifier.migration_classes(proof_path, source)
    candidate = copy.deepcopy(value)
    for name, index, previous in relocations:
        candidate[name][index]['producer'] = current
    candidate['reader_relocation'] = {
        'schema': 'natlang.classification-producer-reader-relocation/1',
        'original': original, 'current_producer': current,
        'historical_producers': list(checked.values()),
        'rows_changed': 0, 'optimizer_updates': 0, 'training_credit': 0}
    raw = (json.dumps(candidate, sort_keys=True, separators=(',', ':')) + '\n').encode()
    directory = Path(directory) / 'classification-reader-relocations'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (classifier.digest(candidate) + '.json')
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('classification reader relocation artifact changed')
    else:
        write_bytes_durable(path, raw)
    result = classifier.migration_classes(path, source)
    # Preserve exact original provenance and detect source changes during use.
    if classifier._binding(proof_path) != original:
        raise ValueError('original classification proof changed during reader relocation')
    if classifier._binding(classifier.__file__) != current:
        raise ValueError('current classification producer changed during reader relocation')
    for previous in checked.values():
        classifier._checked(previous)
    return result
