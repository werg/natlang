#!/usr/bin/env python3
"""Physically inherit a full runtime, install a qualified overlay and relocate controls.

No process launch or readiness is performed. Run on the actual source host with
explicit donor package identity and bound nonpackage dependencies.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training/neuralese'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clone_runtime_tree import clone_runtime_tree
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.native_resident import code_inventory
from compose_native_actor_runtime import bound
from verify_native_actor_runtime import verify_files

PACKAGE = Path('training/neuralese/natlang_neuralese')


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file_hex(path), 'bytes': path.stat().st_size}


def install(overlay, donor, donor_code, dependencies, runtime, controls):
    overlay, donor = Path(overlay).resolve(strict=True), Path(donor).resolve(strict=True)
    runtime = Path(runtime).resolve()
    controls = {role: Path(path).resolve() for role, path in controls.items()}
    if set(controls) != {'lfm', 'mellum'} or len(set(controls.values())) != 2:
        raise ValueError('two distinct explicit production control roots required')
    roots = [runtime, *controls.values()]
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError('runtime and control roots must be separate')
    if any(path.is_relative_to(source) or source.is_relative_to(path)
           for path in roots for source in (overlay, donor)):
        raise ValueError('destinations must not contain or modify donor/overlay sources')
    if any(path.exists() or path.is_symlink() for path in roots):
        raise FileExistsError('immutable runtime/control destination already exists')
    record = verify_files(overlay)
    original = Path(record.get('actor_runtime', str(overlay)))
    if not original.is_absolute() or original == Path('/'):
        raise ValueError('overlay needs its exact recorded absolute original root')
    if canonical_json_sha256_hex(code_inventory(donor/PACKAGE)) != donor_code:
        raise ValueError('actual complete donor package identity changed')
    references = json.loads(Path(dependencies).read_bytes())
    if not references or not isinstance(references, list):
        raise ValueError('explicit inherited dependency bindings required')
    required = donor/'scripts/coreweave/provider_dependencies.py'
    if not any(Path(ref['path']).resolve() == required for ref in references):
        raise ValueError('publication dependency producer must be bound explicitly')
    for reference in references:
        if not bound(reference).is_relative_to(donor):
            raise ValueError('inherited dependency is outside actual donor')
    # No hardlinks, no mutation of the donor, no reuse of an old destination.
    clone_runtime_tree(donor, runtime)
    shutil.rmtree(runtime/PACKAGE)
    shutil.copytree(overlay/PACKAGE, runtime/PACKAGE,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    owned = {}
    for name in record['files']:
        source = overlay/name
        if name.startswith('training/neuralese/natlang_neuralese/'):
            owned[source] = runtime/name
        elif name.startswith('role-control/'):
            _, role, relative = name.split('/', 2)
            target = controls[role]/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            owned[source] = target
        else:
            target = runtime/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            owned[source] = target
    def local_source(path):
        if path in owned:
            return path
        if path.is_absolute() and path.is_relative_to(original):
            candidate = overlay/path.relative_to(original)
            if candidate in owned:
                return candidate
            raise ValueError('original-root metadata names an unrecorded overlay member')
        return None
    def relocate(value):
        if isinstance(value, str):
            path = Path(value)
            # Directory references also relocate, but only from the actual or
            # exact recorded artifact root, never an arbitrary path alias.
            anchor = original if path.is_absolute() and path.is_relative_to(original) else overlay
            if path.is_absolute() and path.is_relative_to(anchor):
                relative = path.relative_to(anchor)
                if len(relative.parts) >= 2 and relative.parts[0] == 'role-control':
                    return str(controls[relative.parts[1]]/Path(*relative.parts[2:]))
                return str(runtime/relative)
            return value
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            result = {key: relocate(item) for key, item in value.items()}
            if {'path', 'sha256'} <= value.keys():
                source = local_source(Path(value['path']))
                if source is None:
                    return result
                # Authenticate transported bytes locally; never reopen the
                # construction host's original absolute path.
                actual_source = binding(source)
                if any(actual_source[key] != value[key] for key in ('sha256', 'bytes') if key in value):
                    raise ValueError('transported overlay metadata binding changed')
                rewrite(source)
                current = binding(owned[source])
                result.update({key: current[key] for key in ('path', 'sha256')})
                if 'bytes' in value:
                    result['bytes'] = current['bytes']
            return result
        return value
    active, done = set(), set()
    def rewrite(source):
        if source in done or source.suffix != '.json':
            return
        if source in active:
            raise ValueError('cyclic bound JSON metadata')
        active.add(source)
        write_json_durable(owned[source], relocate(json.loads(source.read_bytes())))
        active.remove(source)
        done.add(source)
    for source in owned:
        rewrite(source)
    installed = relocate(copy.deepcopy(record))
    installed['files'] = {str(path.relative_to(runtime)): {key: binding(path)[key] for key in ('sha256', 'bytes')}
                          for path in owned.values() if path.is_relative_to(runtime)}
    installed['physical_layout'] = {
        'schema': 'natlang.native-actor-physical-layout/1', 'donor': str(donor),
        'donor_code': donor_code, 'overlay_original_root': str(original), 'overlay_construction': binding(overlay/'construction.json'),
        'runtime': str(runtime), 'control_roots': {role: str(path) for role, path in controls.items()},
        'inherited_dependencies': [], 'replaced_dependencies': [],
        'full_tree_content_inventory_claimed': False,
        'owner_started': False, 'readiness_granted': False}
    for reference in references:
        bound(reference)
        relative = Path(reference['path']).relative_to(donor)
        target = runtime/relative
        actual = binding(target)
        replacement = record['files'].get(str(relative))
        if replacement is not None:
            if any(actual[key] != replacement[key] for key in ('sha256', 'bytes')):
                raise ValueError('sealed dependency replacement differs after overlay')
            installed['physical_layout']['replaced_dependencies'].append(
                {'source': reference, 'replacement': actual})
        else:
            if any(actual[key] != reference[key] for key in ('sha256', 'bytes') if key in reference):
                raise ValueError('inherited dependency differs after physical clone')
            installed['physical_layout']['inherited_dependencies'].append(actual)
    if code_inventory(runtime/PACKAGE) != record['inventory']:
        raise ValueError('qualified numerical package changed during layout')
    write_json_durable(runtime/'construction.json', installed)
    verify_files(runtime)
    return installed


PUBLICATION_PROBE = r'''
import inspect,json,pathlib,runpy,sys
root=pathlib.Path(sys.argv[1]).resolve()
from scripts.coreweave import provider_dependencies
from natlang_neuralese.model import native_checkpoint_control as control
assert pathlib.Path(provider_dependencies.__file__).resolve()==root/'scripts/coreweave/provider_dependencies.py'
assert pathlib.Path(control.__file__).resolve().is_relative_to(root)
record=json.loads((root/'construction.json').read_bytes())
controller_imports={}
for role,reference in record['role_materials'].items():
 material=json.loads(pathlib.Path(reference['path']).read_bytes())
 binder=pathlib.Path(material['binder']['path'])
 scope=runpy.run_path(str(binder),run_name='layout_binder_metadata')
 assert scope['H']==binder.parent
 controller=pathlib.Path(material['controller']['path'])
 assert controller.parent==binder.parent
 assert callable(runpy.run_path(str(controller),run_name='layout_controller_metadata')['main'])
 controller_imports[role]={'control_root':str(binder.parent),'binder_config_path':str(binder.parent/'binder-config.json'),'owner_bound_preflight_performed':False}
selection=control.publication_selection()
control.validate_publication_selection(selection)
parameters=control.publication_trainer_parameters(selection)
assert parameters and callable(provider_dependencies.dependency_closure)
print(json.dumps({'controller_imports':controller_imports,'publication_selection':selection,'trainer_parameters':parameters,
 'dependency_closure_signature':str(inspect.signature(provider_dependencies.dependency_closure)),
 'model_allocations':0,'checkpoint_decodes':0,'owner_started':False,'readiness_granted':False}))
'''


def publication_preflight(runtime):
    runtime = Path(runtime).resolve(strict=True)
    environment = {**os.environ, 'PYTHONPATH': str(runtime/'training/neuralese')+':'+str(runtime),
                   'NATLANG_REPO': str(runtime), 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}
    result = subprocess.run([sys.executable, '-c', PUBLICATION_PROBE, str(runtime)],
                            cwd=runtime, env=environment, text=True, capture_output=True)
    if result.returncode:
        raise ValueError('actual inherited publication preflight failed: '+result.stderr[-5000:])
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--donor', type=Path, required=True)
    parser.add_argument('--donor-code', required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--lfm-control', type=Path, required=True)
    parser.add_argument('--mellum-control', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if any(args.out.resolve().is_relative_to(root.resolve()) for root in
           (args.runtime, args.lfm_control, args.mellum_control)):
        raise ValueError('operation receipt must be outside immutable destination roots')
    if args.out.exists():
        raise FileExistsError('layout operation receipt already exists')
    operation = {'schema': 'natlang.native-actor-layout-install-operation/1',
                 'runtime': str(args.runtime), 'overlay': str(args.overlay), 'donor': str(args.donor),
                 'control_roots': {'lfm': str(args.lfm_control), 'mellum': str(args.mellum_control)},
                 'owner_started': False, 'readiness_granted': False}
    write_json_durable(args.out, {**operation, 'status': 'in_progress'})
    try:
        record = install(args.overlay, args.donor, args.donor_code, args.dependencies,
                         args.runtime, {'lfm': args.lfm_control, 'mellum': args.mellum_control})
        from verify_native_actor_runtime import verify
        proof = {**operation, 'status': 'passed', 'layout': record['physical_layout'],
                 'controls': verify(args.runtime), 'publication': publication_preflight(args.runtime)}
        write_json_durable(args.out, proof)
    except BaseException as error:
        write_json_durable(args.out, {**operation, 'status': 'failed',
                                    'error_type': type(error).__name__, 'error': str(error),
                                    'partial_destinations_must_be_reconciled_before_retry': True})
        raise
    print(json.dumps({'passed': True, 'runtime': str(args.runtime), 'code': record['code'],
                      'receipt': binding(args.out), 'readiness_granted': False}))


if __name__ == '__main__':
    main()
