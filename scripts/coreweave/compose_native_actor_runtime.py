#!/usr/bin/env python3
"""Compose reviewed native inference and coordinator support over an exact joint runtime.

Local artifact construction only: never imports SDK credentials, starts models,
changes a cloud package or creates readiness/training credit.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training/neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex
from natlang_neuralese.common.jsonio import write_bytes_durable, write_json_durable
from natlang_neuralese.train.native_resident import code_inventory

PREFIX = 'training/neuralese/natlang_neuralese/'
RENDERING_ASSETS = tuple('training/prompt-pieces/' + name for name in (
    'pieces.json', 'rendering-contract-fixtures.json', 'rendering-identity-equivalence.json'))
COMMITS = ('e97599138', 'f63c8b05b', '8b4e3bf42', 'fd4b71cbf', 'cb46b28de', 'd1ef19931', '810e496c1', '5fd55d0c7')
ALLOWED = {'model/shared_state.py', 'model/native_adapters.py', 'maple/fused_moe.py',
           'common/runtime_inventory.py', 'rl/production.py', 'rl/native_raw_rounds.py',
           'rl/native_rounds.py', 'serve/__init__.py'}
MECHANISMS = ('rl/native_role_release.py', 'rl/native_raw_rounds.py', 'train/native_readiness.py',
              'eval/native_gradients.py', 'rl/native_raw_policy.py', 'rl/native_raw_handoff.py')



FUSED_UNITS = ('_grouped_dense_mm', '_dense_mm', 'invariant_dense_experts', 'invariant_linear',
               '_invariant_forward_plan', '_invariant_row_block', '_invariant_linear_forward',
               '_InvariantLinear.forward')


def source_unit(raw, name):
    """Exact reviewed definition bytes, including its decorators."""
    nodes = ast.parse(raw).body
    for part in name.split('.'):
        node = next((node for node in nodes if getattr(node, 'name', None) == part), None)
        if node is None:
            return None
        nodes = getattr(node, 'body', ())
    start = min([node.lineno, *[item.lineno for item in getattr(node, 'decorator_list', ())]]) - 1
    return start, node.end_lineno, ''.join(raw.splitlines(True)[start:node.end_lineno])


def rebase_fused_inference(original, reviewed):
    """Keep exact R15 adjoints; transplant only byte-exact reviewed forward units."""
    result = original
    for name in FUSED_UNITS:
        new = source_unit(reviewed, name)
        if new is None:raise ValueError('reviewed inference unit missing: '+name)
        old = source_unit(result, name)
        if old is None:
            result += '\n\n' + new[2] + '\n'
        else:
            lines = result.splitlines(True)
            result = ''.join(lines[:old[0]]) + new[2] + ''.join(lines[old[1]:])
    for name in ('_InvariantLinear.backward', '_DenseExpertProjection', 'invariant_weight_gradient'):
        if source_unit(result,name)[2] != source_unit(original,name)[2]:
            raise ValueError('inference composition changed R15 adjoints: '+name)
    for name in FUSED_UNITS:
        if source_unit(result,name)[2] != source_unit(reviewed,name)[2]:
            raise ValueError('inference composition differs from its reviewed source: '+name)
    return result


def bound(reference):
    path = Path(reference['path']).resolve(strict=True)
    if (sha256_file_hex(path) != reference['sha256']
            or ('bytes' in reference and path.stat().st_size != reference['bytes'])):
        raise ValueError('composition source binding changed: '+str(path))
    return path


def include_rendering_assets(source, destination):
    """Carry the exact non-Python inputs used by the actor's rendering identity."""
    for name in RENDERING_ASSETS:
        path = Path(source) / name
        raw = path.read_bytes()
        target = Path(destination) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        write_bytes_durable(target, raw)


def controller_publishing_closure(raw):
    """Use the existing exact publisher loader for the controller's same closure."""
    start = raw.index(' # Explicit external control closure')
    stop = raw.index(' from natlang_neuralese.common.jsonio import write_json_durable as write', start)
    raw = raw[:start] + ' control_load=load_controls(spec)\n' + raw[stop:]
    helper = 'def load_controls(spec):\n """Load exactly the same reviewed control actors used by role publication."""\n loader=bound(spec[\'publisher_loader\'])\n if spec[\'publisher_loader\'][\'sha256\']!=\'03c432696073458523a9ae3e7f2f4e1d8f4a766dff25c21eb6493c46b03467d8\':\n  raise ValueError(\'Controller requires its reviewed shared publishing loader\')\n manifest=json.loads(bound(spec[\'publisher_manifest\']).read_bytes())\n if (manifest[\'base_loader\']!=spec[\'control_loader\'] or manifest[\'base_manifest\']!=spec[\'control_manifest\']\n     or manifest[\'controls\']!=spec[\'controls\'] or Path(manifest[\'package\'])!=Path(spec[\'numerical_package\'])\n     or manifest[\'code_sha256\']!=spec[\'numerical_code_sha256\'] or manifest[\'role\']!=spec[\'model\']):\n  raise ValueError(\'Controller and publisher must bind the same complete role closure\')\n scope={\'__file__\':str(loader),\'__name__\':\'owned_controller_publishing_loader\'}\n exec(compile(loader.read_bytes(),str(loader),\'exec\'),scope)\n return scope[\'load\'](spec[\'publisher_manifest\'])\n\n\n'
    return raw.replace('def main():', helper + 'def main():', 1)


def compose(base_construction, base_sha256, source, out, closures, publisher_loader, controller=None, binder=None):
    base_construction, source, out = Path(base_construction), Path(source), Path(out).resolve()
    base_reference = {'path': str(base_construction.resolve()), 'sha256': base_sha256}
    record = json.loads(bound(base_reference).read_bytes())
    base = base_construction.parent/PREFIX
    if (code_inventory(base) != record['inventory'] or digest(record['inventory']) != record['code']):
        raise ValueError('joint runtime package differs from its exact construction')
    if set(closures) != {'lfm', 'mellum'}:
        raise ValueError('future role control needs both explicit source closures')
    if out.exists():
        raise FileExistsError('immutable composition already exists: '+str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix='.'+out.name+'-', dir=out.parent))
    package = pending/PREFIX
    try:
        package.parent.mkdir(parents=True)
        shutil.copytree(base, package, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if code_inventory(package) != record['inventory']:
            raise ValueError('base source changed while composing')
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=source)
        applied = []
        for commit in COMMITS:
            commit = git('rev-parse', commit+'^{commit}').decode().strip()
            paths = git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).decode().splitlines()
            paths = [name for name in paths if name.startswith(PREFIX) and name.endswith('.py')]
            fused_path = PREFIX+'maple/fused_moe.py'
            reviewed_paths = paths.copy()
            paths = [name for name in paths if name != fused_path]
            if not reviewed_paths or not {name.removeprefix(PREFIX) for name in reviewed_paths} <= ALLOWED:
                raise ValueError('reviewed patch escaped the declared inference/controller scope')
            patch = git('diff', commit+'^', commit, '--', *paths)
            merged = {}
            # Ordinary three-way git merge preserves independently composed R15
            # changes even where an old file has different blank-line context.
            with tempfile.TemporaryDirectory(prefix='native-runtime-merge-') as merge_dir:
                for name in paths:
                    target = pending/name
                    before = target.read_bytes() if target.exists() else b''
                    ancestor = git('show', commit+'^:'+name)
                    changed = git('show', commit+':'+name)
                    inputs = [Path(merge_dir)/label for label in ('current','ancestor','reviewed')]
                    for path, raw in zip(inputs,(before,ancestor,changed)):
                        path.write_bytes(raw)
                    check = subprocess.run(['git','merge-file','-p',*[str(path) for path in inputs]],
                                           cwd=source,capture_output=True)
                    if check.returncode:
                        raise ValueError('reviewed three-way merge conflicted: '+commit+' '+name)
                    write_bytes_durable(target,check.stdout)
                    merged[name] = {'before':hashlib.sha256(before).hexdigest(),
                                    'after':hashlib.sha256(check.stdout).hexdigest()}
            applied.append({'commit': commit, 'paths': paths, 'reviewed_paths': reviewed_paths, 'patch_sha256': hashlib.sha256(patch).hexdigest(),
                            'three_way_files': merged})
        fused_path = package/'maple/fused_moe.py'
        original_fused = fused_path.read_text()
        reviewed_fused = git('show', 'cb46b28de:'+PREFIX+'maple/fused_moe.py').decode()
        write_bytes_durable(fused_path,rebase_fused_inference(original_fused,reviewed_fused).encode())
        inventory = code_inventory(package)
        changes = {name for name in inventory.keys() | record['inventory'].keys()
                   if inventory.get(name) != record['inventory'].get(name)}
        if changes != ALLOWED:
            raise ValueError('composition changed unexpected or missing modules: '+repr(changes))
        entry = 'scripts/neuralese_data/serve_direct_checkpoint.py'
        destination = pending/entry
        destination.parent.mkdir(parents=True)
        write_bytes_durable(destination, git('show', 'd1ef19931:'+entry))
        coordinator_scripts = ('scripts/run_bonsai_queue.py',
            'scripts/coreweave/teacher_engine_guard.py',
            'scripts/coreweave/teacher_generation_guardian.py',
            'scripts/coreweave/teacher_batch_completion.py')
        for name in coordinator_scripts:
            target = pending/name
            target.parent.mkdir(parents=True,exist_ok=True)
            write_bytes_durable(target,git('show','d1ef19931:'+name))
        # These helpers run on the real source host where /work bindings exist.
        # Keep their sibling imports self-contained and pinned to reviewed bytes.
        utility_commit = git('rev-parse', 'HEAD').decode().strip()
        for name in ('compose_native_actor_runtime.py', 'prepare_native_actor_role.py',
                     'verify_native_actor_runtime.py'):
            target = pending/'scripts/coreweave'/name
            write_bytes_durable(target,git('show',utility_commit+':scripts/coreweave/'+name))
        # Transport the installer and its existing physical-clone companion as
        # exact source files too. No dependence on a supervisor worktree remains.
        layout_tools_commit = git('rev-parse', 'HEAD').decode().strip()
        for name in ('scripts/coreweave/install_native_actor_layout.py', 'scripts/clone_runtime_tree.py'):
            target = pending/name
            target.parent.mkdir(parents=True,exist_ok=True)
            write_bytes_durable(target,git('show',layout_tools_commit+':'+name))
        def reference(path):
            path = Path(path)
            return {'path': str(out/path.relative_to(pending)),
                    'sha256': sha256_file_hex(path), 'bytes': path.stat().st_size}
        def copied(reference_value, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            write_bytes_durable(destination, bound(reference_value).read_bytes())
            return reference(destination)
        authority, control_records = {}, {}
        for role, closure_reference in closures.items():
            closure = json.loads(bound(closure_reference).read_bytes())
            if closure.get('role') != role:
                raise ValueError('selected control source belongs to another role')
            control_root = pending/'role-control'/role
            control_root.mkdir(parents=True)
            loader = copied(closure['loader'], control_root/'control-loader.py')
            manifest = copy.deepcopy(json.loads(bound(closure['manifest']).read_bytes()))
            manifest['package'] = str(out/PREFIX)
            manifest['base_code_sha256'] = digest(inventory)
            sources = {}
            for item in manifest['modules']:
                name = item['module'].replace('.', '/')+'.py'
                item['source'] = copied(item['source'], control_root/'base'/name)
                item['installed_sha256'] = inventory[name]
                sources[name] = item['source']['sha256']
            write_json_durable(control_root/'control-manifest.json', manifest)
            controls = {}
            for name, old in closure['controls'].items():
                destination = control_root/'extra'/(name+'.py')
                controls[name] = copied(old, destination)
                if name == 'native_supervised_boundary':
                    # Actual R14 material already uses this reviewed task-runtime
                    # fix; the older host closure is source provenance, not a
                    # reason to regress the next controller's collector binding.
                    write_bytes_durable(destination,git('show',
                        '5778b3422:'+PREFIX+'rl/native_supervised_boundary.py'))
                    controls[name] = reference(destination)
                if name == 'native_raw_rounds':
                    # Apply the same reviewed one-line bootstrap plumbing to the
                    # final active external override, not a discarded base copy.
                    patch = git('diff', 'd1ef19931^', 'd1ef19931', '--', PREFIX+'rl/native_raw_rounds.py')
                    scratch = control_root/'patch'
                    target = scratch/PREFIX/'rl/native_raw_rounds.py'
                    target.parent.mkdir(parents=True)
                    write_bytes_durable(target, destination.read_bytes())
                    subprocess.run(['git','apply','--unsafe-paths','--directory='+str(scratch)],
                                   input=patch,cwd=source,check=True)
                    write_bytes_durable(destination,target.read_bytes());shutil.rmtree(scratch)
                    controls[name] = reference(destination)
                sources['rl/'+name+'.py'] = controls[name]['sha256']
            authority[role] = {name: sources.get(name, inventory[name]) for name in MECHANISMS}
            control_records[role] = {'loader': loader,
                'manifest': reference(control_root/'control-manifest.json'), 'controls': controls,
                'source_closure': closure_reference}
        write_json_durable(pending/'role-mechanism-authorities.json', authority)
        authority_reference = reference(pending/'role-mechanism-authorities.json')
        loader_reference = copied(publisher_loader, pending/'publishing-loader.py')
        from natlang_neuralese.train.native_source_classification import source_policy
        for role, closure in control_records.items():
            manifest = {'schema':'natlang.source-host-publishing-control-closure/1',
                'role':role,'package':str(out/PREFIX),'code_sha256':digest(inventory),
                'base_loader':closure['loader'],'base_manifest':closure['manifest'],
                'controls':closure['controls'],'role_authorities':authority_reference,
                'source_policy_sha256':source_policy(package)}
            path = pending/'role-control'/role/'publishing-manifest.json'
            write_json_durable(path,manifest)
            closure['publisher_manifest'] = reference(path)
        materials = {}
        if (controller is None) != (binder is None):
            raise ValueError('future binder material requires both existing reviewed sources')
        if controller is not None:
            original_binder = bound(binder).read_text()
            replacements = {
                "future['owned_models']==['mellum']": "future['owned_models']==[config['model']]",
                "if x['model']=='mellum'": "if x['model']==config['model']",
                "model='mellum',entry_root": "model=config['model'],entry_root",
                "controls=config['controls'],bootstrap=":
                    "controls=config['controls'],publisher_loader=config['publisher_loader'],publisher_manifest=config['publisher_manifest'],bootstrap=",
                "  task_runtime=config.get('task_runtime')":
                    "  declaration['actor_runtime']=config['actor_runtime']\n  task_runtime=config.get('task_runtime')",
                "assert root_review['passed']and root_review['controller']==config['controller']and root_review['binder']==binding(__file__)":
                    "assert root_review['passed'] is True and all(root_review['controller'].get(k)==config['controller'][k] for k in ('path','sha256')) and all(root_review['binder'].get(k)==binding(__file__)[k] for k in ('path','sha256'));bound(root_review['controller']);bound(root_review['binder'])",
            }
            for old, new in replacements.items():
                if original_binder.count(old) != 1:
                    raise ValueError('reviewed binder source shape changed: '+old)
                original_binder = original_binder.replace(old,new,1)
            for role, controls in control_records.items():
                folder=pending/'role-control'/role
                controller_reference=copied(controller,folder/'controller.py')
                write_bytes_durable(folder/'controller.py',controller_publishing_closure(
                    (folder/'controller.py').read_text()).encode())
                controller_reference=reference(folder/'controller.py')
                write_bytes_durable(folder/'bind-controller.py',original_binder.encode())
                material={'schema':'natlang.role-aware-controller-control-material/1',
                    'model':role,'runtime':str(out),'runtime_code_sha256':digest(inventory),
                    'actor_runtime':str(out),'controller':controller_reference,
                    'binder':reference(folder/'bind-controller.py'),
                    'publisher_loader':loader_reference,'publisher_manifest':controls['publisher_manifest'],
                    'control_loader':controls['loader'],'control_manifest':controls['manifest'],
                    'controls':controls['controls'],'role_mechanism_authorities':authority_reference,
                    'requires_verified_later_runtime_handoff':True,
                    'old_owner_or_source_plan_relabeling_allowed':False}
                write_json_durable(folder/'control-material.json',material)
                materials[role]=reference(folder/'control-material.json')
        include_rendering_assets(source, pending)
        for path in pending.rglob('*.py'):
            compile(path.read_bytes(),str(path),'exec')
        files = {str(path.relative_to(pending)): {'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
                 for path in sorted(pending.rglob('*')) if path.is_file()}
        result = {'schema':'natlang.native-actor-coordinator-composition/1',
            'base':base_reference,'base_code':record['code'],'code':digest(inventory),
            'inventory':inventory,'files':files,'applied':applied,'changed_modules':sorted(changes),
            'fused_inference_rebase':{'reviewed_commit':'cb46b28de','exact_units':list(FUSED_UNITS),
                'r15_adjoints_byte_unchanged':True,'unrelated_deferred_gradient_machinery_imported':False},
            'role_controls':control_records,'role_mechanism_authorities':authority,
            'preparation_utilities_commit':utility_commit,'layout_tools_commit':layout_tools_commit,
            'external_control_commits':{'native_supervised_boundary':'5778b3422'},
            'role_materials':materials,'controller_source':controller,'binder_source':binder,
            'publisher_loader':loader_reference,'actor_entrypoint':reference(pending/entry),
            'actor_runtime':str(out),'qualification_credit':False,'cross_model_readiness_granted':False,
            'cloud_mutations':0,'model_allocations':0,
            'six_mechanism_authority_scope':'future composed controllers only; not historic owner proofs'}
        write_json_durable(pending/'construction.json',result)
        pending.rename(out)
        descriptor=os.open(out.parent,os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        return result
    except BaseException:
        shutil.rmtree(pending,ignore_errors=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-construction',type=Path,required=True)
    parser.add_argument('--base-sha256',required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--control-closure',action='append',required=True,help='role=exact binding JSON file')
    parser.add_argument('--publisher-loader',type=Path,required=True)
    parser.add_argument('--controller',type=Path)
    parser.add_argument('--binder',type=Path)
    args=parser.parse_args()
    closures={}
    for value in args.control_closure:
        role,path=value.split('=',1);path=Path(path).resolve(strict=True)
        if role in closures:parser.error('duplicate role closure')
        closures[role]={'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
    publisher=args.publisher_loader.resolve(strict=True)
    def source_binding(path):
        if path is None:return None
        path=path.resolve(strict=True)
        return {'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
    result=compose(args.base_construction,args.base_sha256,args.source,args.out,closures,
        {'path':str(publisher),'sha256':sha256_file_hex(publisher),'bytes':publisher.stat().st_size},
        source_binding(args.controller),source_binding(args.binder))
    print(json.dumps({'construction':str(args.out/'construction.json'),'code':result['code'],
                      'files':len(result['files']),'cloud_mutations':0,'model_allocations':0}))


if __name__=='__main__':main()
