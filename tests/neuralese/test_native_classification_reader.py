"""Actual migration checks retain historical bytes across a reader relocation."""
import json
from pathlib import Path

import pytest

from natlang_neuralese.train import native_source_classification as classifier
from natlang_neuralese.train.native_classification_reader import migration_classes
from test_native_source_classification import fixture


@pytest.mark.parametrize('damage', [None, 'changed-code', 'missing-origin', 'wrong-bytes', 'wrong-source', 'changed-index'])
def test_actual_relocated_classifier_checks_all_original_custody(tmp_path, monkeypatch, damage):
    source, old, old_package, package, current = fixture(tmp_path, monkeypatch)
    part = classifier.migrate_index(source, old, old_package=old_package, new_package=package)
    proof = {'schema': classifier.SCHEMA, 'train_sha256': current.receipt()['sha256'], 'indexes': [part]}
    path = tmp_path / 'proof.json';path.write_text(json.dumps(proof))
    expected = classifier.migration_classes(path, current)
    origin = tmp_path / 'original-runtime/train/native_source_classification.py'
    origin.parent.mkdir(parents=True);origin.write_bytes(Path(classifier.__file__).read_bytes())
    part['producer'] = classifier._binding(origin)
    if damage == 'changed-code':
        origin.write_bytes(origin.read_bytes() + b'\n# changed producer\n');part['producer'] = classifier._binding(origin)
    elif damage == 'missing-origin':origin.unlink()
    elif damage == 'wrong-bytes':part['producer']['bytes'] += 1
    elif damage == 'wrong-source':proof['train_sha256'] = '0' * 64
    elif damage == 'changed-index':
        index = Path(part['old_index']['path']);index.write_bytes(index.read_bytes() + b'changed')
    path.write_text(json.dumps(proof));before = path.read_bytes()
    if damage:
        with pytest.raises((ValueError, FileNotFoundError)):
            migration_classes(path, current, directory=tmp_path / 'out')
    else:
        result = migration_classes(path, current, directory=tmp_path / 'out')
        assert result == expected
        assert migration_classes(path, current, directory=tmp_path / 'out') == expected
        aliases = list((tmp_path / 'out/classification-reader-relocations').glob('*.json'))
        assert len(aliases) == 1
        derived = json.loads(aliases[0].read_bytes())
        assert derived['reader_relocation']['original'] == classifier._binding(path)
        assert derived['reader_relocation']['historical_producers'] == [classifier._binding(origin)]
        assert derived['reader_relocation']['training_credit'] == 0
        assert derived['indexes'][0]['producer'] == classifier._binding(classifier.__file__)
        aliases[0].write_text('{}')
        with pytest.raises(ValueError, match='artifact changed'):
            migration_classes(path, current, directory=tmp_path / 'out')
    assert path.read_bytes() == before
