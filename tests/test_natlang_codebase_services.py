import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rl_tasks.codebase_services import load_external_services


def _spec(root: Path, source: str):
    path = root / 'services' / 'inspection-service.ts'
    path.parent.mkdir(parents=True)
    path.write_text(source)
    return {'external_services': {'inspections': {
        'source_path': 'services/inspection-service.ts',
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }}}


def test_loads_hash_bound_service_outside_editable_package(tmp_path: Path):
    root = tmp_path / 'codebase'
    package = root / 'package'
    package.mkdir(parents=True)
    source = 'export async function sourceDigest(): Promise<string> { return "abc"; }\n'
    metadata = _spec(root, source)
    services, bindings = load_external_services(package, metadata)
    assert services == {'inspections': source}
    assert bindings == {'inspections': {'source_path': 'services/inspection-service.ts',
                                        'source_sha256': hashlib.sha256(source.encode()).hexdigest()}}


def test_rejects_hash_mismatch(tmp_path: Path):
    root = tmp_path / 'codebase'
    package = root / 'package'
    package.mkdir(parents=True)
    metadata = _spec(root, 'export function read(): number { return 1; }\n')
    metadata['external_services']['inspections']['source_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash mismatch'):
        load_external_services(package, metadata)


def test_rejects_service_source_inside_model_editable_package(tmp_path: Path):
    package = tmp_path / 'codebase' / 'package'
    package.mkdir(parents=True)
    with pytest.raises(ValueError, match='outside'):
        load_external_services(package, {'external_services': {'inspections': {
            'source_path': 'package/service.ts', 'source_sha256': '0' * 64
        }}})
