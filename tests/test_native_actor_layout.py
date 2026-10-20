"""Physical-layout metadata guards; no models, checkpoints or cloud resources."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/coreweave'))
from install_native_actor_layout import PACKAGE, binding, install
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
              'role_materials': {role: binding(overlay/'role-control'/role/'control-material.json')
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


def test_layout_rejects_changed_donor_before_claiming_destination(tmp_path):
    arguments = fixture_layout(tmp_path)
    _, donor, _, _, runtime, controls = arguments
    (donor/'scripts/coreweave/provider_dependencies.py').write_text('changed')
    with pytest.raises(ValueError, match='binding changed'):
        install(*arguments)
    assert not runtime.exists()
    assert not any(root.exists() for root in controls.values())


def test_layout_refuses_existing_control_root(tmp_path):
    arguments = fixture_layout(tmp_path)
    arguments[-1]['lfm'].mkdir(parents=True)
    with pytest.raises(FileExistsError):
        install(*arguments)
    assert not arguments[-2].exists()
