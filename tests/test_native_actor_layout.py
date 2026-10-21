"""Physical-layout metadata guards; no models, checkpoints or cloud resources."""
import json
import shutil
import os
import subprocess
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/coreweave'))
from install_native_actor_layout import PACKAGE, binding, install
from compose_native_actor_runtime import RENDERING_ASSETS, include_rendering_assets
from verify_native_actor_runtime import verify_files
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.train.native_resident import code_inventory


def fixture_layout(tmp_path):
    donor, overlay = tmp_path/'donor', tmp_path/'overlay'
    for root in (donor, overlay):
        package = root/PACKAGE
        package.mkdir(parents=True)
        (package/'__init__.py').write_text('VALUE = 1\n')
    dependency = donor/'scripts/coreweave/provider_dependencies.py'
    dependency.parent.mkdir(parents=True)
    dependency.write_text('RETAINED = True\n')
    references = tmp_path/'dependencies.json'
    references.write_text(json.dumps([binding(dependency)]))
    controls = {role: tmp_path/'control'/role for role in ('lfm', 'mellum')}
    for role in controls:
        folder = overlay/'role-control'/role
        folder.mkdir(parents=True)
        (folder/'bind-controller.py').write_text('from pathlib import Path\nH=Path(__file__).resolve().parent\n')
        (folder/'control-manifest.json').write_text(json.dumps({'package': str(overlay/PACKAGE)}))
        (folder/'control-material.json').write_text(json.dumps({
            'runtime': str(overlay), 'binder': binding(folder/'bind-controller.py'),
            'control_manifest': binding(folder/'control-manifest.json')}))
    inventory = code_inventory(overlay/PACKAGE)
    files = {str(path.relative_to(overlay)): {key: binding(path)[key] for key in ('sha256', 'bytes')}
             for path in overlay.rglob('*') if path.is_file()}
    record = {'schema': 'natlang.native-actor-coordinator-composition/1', 'files': files,
              'inventory': inventory, 'code': canonical_json_sha256_hex(inventory),
              'actor_runtime': str(overlay), 'role_materials': {role: binding(overlay/'role-control'/role/'control-material.json')
                                 for role in controls}}
    (overlay/'construction.json').write_text(json.dumps(record))
    return overlay, donor, canonical_json_sha256_hex(inventory), references, tmp_path/'runtime', controls


def test_layout_rebinds_json_dependency_hashes_and_binder_parent(tmp_path):
    arguments = fixture_layout(tmp_path)
    overlay, donor, _, _, runtime, controls = arguments
    original = (overlay/'role-control/lfm/control-material.json').read_bytes()
    result = install(*arguments)
    assert (runtime/'scripts/coreweave/provider_dependencies.py').read_bytes() == (
        donor/'scripts/coreweave/provider_dependencies.py').read_bytes()
    assert (overlay/'role-control/lfm/control-material.json').read_bytes() == original
    for role, root in controls.items():
        material = json.loads((root/'control-material.json').read_bytes())
        assert Path(material['binder']['path']).parent == root
        assert material['binder'] == binding(root/'bind-controller.py')
        assert material['control_manifest'] == binding(root/'control-manifest.json')
        assert json.loads((root/'control-manifest.json').read_bytes())['package'] == str(runtime/PACKAGE)
        assert result['role_materials'][role] == binding(root/'control-material.json')
    assert result['physical_layout']['readiness_granted'] is False
    assert code_inventory(runtime/PACKAGE) == code_inventory(overlay/PACKAGE)


def refresh_construction(root):
    path = root/'construction.json'
    record = json.loads(path.read_bytes())
    record['inventory'] = code_inventory(root/PACKAGE)
    record['code'] = canonical_json_sha256_hex(record['inventory'])
    record['files'] = {str(p.relative_to(root)): {k: binding(p)[k] for k in ('sha256', 'bytes')}
                       for p in root.rglob('*') if p.is_file() and p != path}
    path.write_text(json.dumps(record))


def test_installed_actor_computes_real_rendering_identity_without_checkout(tmp_path):
    arguments = fixture_layout(tmp_path)
    overlay, _, _, _, runtime, _ = arguments
    source = Path(__file__).resolve().parents[1]
    shutil.rmtree(overlay/PACKAGE)
    shutil.copytree(source/PACKAGE, overlay/PACKAGE,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    include_rendering_assets(source, overlay)
    refresh_construction(overlay)
    install(*arguments)
    code = ('import json;from natlang_neuralese.data.rendering_contract import compute_identity;'
            'print(json.dumps({mode:compute_identity(mode)for mode in ("full","program-header")}))')
    result = subprocess.run([sys.executable, '-P', '-c', code], cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(runtime/'training/neuralese'),
             'NATLANG_REPO': str(runtime), 'CUDA_VISIBLE_DEVICES': ''},
        capture_output=True, text=True, check=True)
    identities = json.loads(result.stdout)
    for mode, identity in identities.items():
        assert identity['mode'] == mode
        assert identity['fixtures_sha256'] == binding(runtime/RENDERING_ASSETS[1])['sha256']
        assert identity['pieces_sha256'] == binding(runtime/RENDERING_ASSETS[0])['sha256']


@pytest.mark.parametrize('asset', RENDERING_ASSETS)
def test_actor_rejects_rendering_assets_missing_from_manifest(tmp_path, asset):
    overlay = fixture_layout(tmp_path)[0]
    module = overlay/PACKAGE/'data/rendering_contract.py'
    module.parent.mkdir()
    module.write_text('VALUE = 1\n')
    include_rendering_assets(Path(__file__).resolve().parents[1], overlay)
    refresh_construction(overlay)
    path = overlay/'construction.json'
    record = json.loads(path.read_bytes())
    del record['files'][asset]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='rendering asset is absent'):
        verify_files(overlay)


def test_layout_rejects_changed_donor_before_claiming_destination(tmp_path):
    arguments = fixture_layout(tmp_path)
    _, donor, _, _, runtime, controls = arguments
    (donor/'scripts/coreweave/provider_dependencies.py').write_text('changed')
    with pytest.raises(ValueError, match='binding changed'):
        install(*arguments)
    assert not runtime.exists()
    assert not any(root.exists() for root in controls.values())


def replacement_fixture(tmp_path):
    arguments = fixture_layout(tmp_path)
    overlay, donor = arguments[:2]
    relative = 'scripts/coreweave/provider_dependencies.py'
    replacement = overlay/relative
    replacement.parent.mkdir(parents=True)
    replacement.write_text('REVIEWED_SUCCESSOR = True\n')
    path = overlay/'construction.json'
    record = json.loads(path.read_bytes())
    record['files'][relative] = {key: binding(replacement)[key] for key in ('sha256', 'bytes')}
    path.write_text(json.dumps(record))
    return arguments, replacement, donor/relative


def test_sealed_dependency_replacement_records_both_actual_sources(tmp_path):
    arguments, replacement, original = replacement_fixture(tmp_path)
    original_bytes = original.read_bytes()
    result = install(*arguments)
    actual = arguments[-2]/replacement.relative_to(arguments[0])
    assert actual.read_bytes() == replacement.read_bytes()
    assert original.read_bytes() == original_bytes
    assert result['physical_layout']['inherited_dependencies'] == []
    assert result['physical_layout']['replaced_dependencies'] == [
        {'source': binding(original), 'replacement': binding(actual)}]
    assert result['physical_layout']['owner_started'] is False


def test_changed_sealed_replacement_is_rejected_before_destination(tmp_path):
    arguments, replacement, _ = replacement_fixture(tmp_path)
    replacement.write_text('UNREVIEWED = True\n')
    with pytest.raises(ValueError, match='composed source file changed'):
        install(*arguments)
    assert not arguments[-2].exists()


def test_replacement_does_not_excuse_changed_original_donor(tmp_path):
    arguments, _, original = replacement_fixture(tmp_path)
    original.write_text('changed source\n')
    with pytest.raises(ValueError, match='binding changed'):
        install(*arguments)
    assert not arguments[-2].exists()


def test_layout_refuses_existing_control_root(tmp_path):
    arguments = fixture_layout(tmp_path)
    arguments[-1]['lfm'].mkdir(parents=True)
    with pytest.raises(FileExistsError):
        install(*arguments)
    assert not arguments[-2].exists()


def test_moved_overlay_never_reads_original_host_paths(tmp_path):
    arguments = list(fixture_layout(tmp_path))
    original = arguments[0]
    moved = tmp_path/'transported-overlay'
    shutil.copytree(original, moved)
    shutil.rmtree(original)
    arguments[0] = moved
    result = install(*arguments)
    assert not original.exists()
    assert result['actor_runtime'] == str(arguments[-2])
    assert result['physical_layout']['overlay_original_root'] == str(original)
    for role, root in arguments[-1].items():
        material = json.loads((root/'control-material.json').read_bytes())
        assert material['binder'] == binding(root/'bind-controller.py')
        assert material['control_manifest'] == binding(root/'control-manifest.json')
        assert result['role_materials'][role] == binding(root/'control-material.json')


def test_failed_cli_leaves_explicit_operation_receipt(tmp_path, monkeypatch):
    import install_native_actor_layout as module
    overlay, donor, code, references, runtime, controls = fixture_layout(tmp_path)
    controls['lfm'].mkdir(parents=True)
    out = tmp_path/'operation.json'
    monkeypatch.setattr(sys, 'argv', ['install-layout', '--overlay', str(overlay),
        '--donor', str(donor), '--donor-code', code, '--dependencies', str(references),
        '--runtime', str(runtime), '--lfm-control', str(controls['lfm']),
        '--mellum-control', str(controls['mellum']), '--out', str(out)])
    with pytest.raises(FileExistsError):
        module.main()
    receipt = json.loads(out.read_bytes())
    assert receipt['status'] == 'failed'
    assert receipt['partial_destinations_must_be_reconciled_before_retry'] is True
    assert receipt['owner_started'] is False and receipt['readiness_granted'] is False
    assert not runtime.exists()
