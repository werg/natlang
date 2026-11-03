#!/usr/bin/env python3
"""Windows of one mandatory joint-SFT continuation (natlang.rl-production-config/1 with mandatory_sft_only).

A mandatory supervised controller trains until its qualified semantic-refresh boundary, then stops with
``phase: awaiting_semantic_refresh`` and ``pending_native_checkpoint``: one *window*. The next window is a new
controller (its own ``out``) continuing from that earned checkpoint with the same recipe, objective, focus,
optimizer/RNG/exposure and scored-graph anchor (native_supervised.continuation_config). An interrupted window resumes
in its own slot through ``rolling_sft_resume``. The same steps serve the DGX and the cloud lineages
(scripts/coreweave/rl_cloud.py, lfm3_rl_lineage.sh).

Run with the package of the runtime that trains the lineage on PYTHONPATH (a frozen runtime; this file only uses
functions that runtime already has):

  sft_window.py next --previous-config CFG --out DIR --config-out PATH [--checkpoint-path P] [--window N] [--id ID]
  sft_window.py rolling --config CFG --config-out PATH
  sft_window.py closure --config CFG -o OUT.json     exact input files a window's controller and learner read
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

GIB = 1024 ** 3
REPO = Path(__file__).resolve().parents[2]
NESTED_JSON_BYTES = 64 * 1024 ** 2
NESTED_JSON_DEPTH = 4
# Bound records above this are census/objective payloads (families, objectives, rolling custody), not admission
# records; shipping them all would add ~2.6 GB per seed.
NESTED_JSON_SHIP_BYTES = 16 * 1024 ** 2


def _require_new(path):
    path = Path(path)
    if path.exists():
        raise SystemExit(f'{path} exists: a prepared control is immutable')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _folded_admissions(training, options):
    history = (options.get('native_shared_execution_history')
               or (training.get('options') or {}).get('native_shared_execution_history') or [])
    return [e for e in history if isinstance(e, dict) and isinstance(e.get('qualification'), dict)]


def next_window(previous_config, out, config_out, *, checkpoint_path=None, window=None, ident=None):
    """The next window from the previous window's actual earned boundary (generalizes the DGX
    prepare-earned-window-continuation operator; no host-only checks or wrappers)."""
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.train.native_resident import binding, bound
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    from natlang_neuralese.train.native_operation_report import supervised_progress_numerics_ok
    from natlang_neuralese.train.recipe import load_recipe
    from natlang_neuralese.train.native_execution_transition import require_qualified_for_training
    from natlang_neuralese.rl import native_supervised as ns

    previous_config = Path(previous_config)
    c = json.loads(previous_config.read_text())
    if not c.get('mandatory_sft_only'):
        raise SystemExit('not a mandatory supervised controller config')
    previous = Path(c['out'])
    state = json.loads((previous / 'state.json').read_text())
    if state.get('phase') != 'awaiting_semantic_refresh':
        raise SystemExit(f'previous window is {state.get("phase")!r}, not at its semantic-refresh boundary')
    parent = Path(state['pending_native_checkpoint']).resolve(strict=True)
    if parent != (previous / 'learner/checkpoint.pt').resolve(strict=True):
        raise SystemExit('pending checkpoint is not the previous window learner slot')
    ref = binding(parent)
    report_path = previous / 'learner/report.json'
    report = json.loads(report_path.read_text())
    if report.get('checkpoint_sha256') != ref['sha256'] or not supervised_progress_numerics_ok(report, ref):
        raise SystemExit('previous window report does not qualify its checkpoint')
    boundary = report.get('continuation_boundary') or {}
    if not (boundary.get('kind') == 'semantic-refresh' and boundary.get('reached') is True
            and boundary.get('new_optimizer_updates', 0) >= 1):
        raise SystemExit('previous window did not reach an earned semantic-refresh boundary')
    if Path(config_out).is_file():
        return adopt_prepared_window(Path(config_out), Path(out), previous_config, parent, ref)
    training = CheckpointPolicy.load(parent)['training']
    if not training.get('optimizer') or not training.get('rng'):
        raise SystemExit('earned boundary lacks optimizer or RNG state')
    stage_id = c['production_stage']
    frame = collect_sft_progress(training, borrow_frames=True)[stage_id]
    options = frame.get('options') or {}
    anchor = copy.deepcopy(options.get('sft_update_budget_reference'))
    folded = _folded_admissions(training, options)
    if folded:
        if len(folded) != 1:
            raise SystemExit('earned boundary carries more than one folded execution admission')
        require_qualified_for_training(training, admission_ref=folded[0]['qualification']['admission'],
                                       checkpoint_path=str(parent))
        # The trained successor carries its folded admission itself; the parent controller's authority is not
        # re-inherited (main 90bbf7fc7 _successor_folded_admission; frozen runtimes predate it).
        ns.inherited_execution_admission = lambda config: None
    receipt = json.loads((previous / 'learner/native-training-receipt.json').read_text())
    if receipt.get('recipe') != c['trainer_parameters']['recipe_binding']:
        raise SystemExit('previous window receipt recipe differs from its control')
    recipe = json.loads(bound({'path': receipt['recipe']['path'], 'sha256': receipt['recipe']['file_sha256']}).read_text())
    stage = next(s for s in recipe['stages'] if s['id'] == stage_id)
    prior_window = stage['parameters'].get('refresh_after_updates')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty')
    out.mkdir(parents=True, exist_ok=True)
    config_out = _require_new(config_out)
    if window is not None and window != prior_window:
        stage['parameters']['refresh_after_updates'] = window
        recipe_path = out / 'recipe-runtime.json'
        write_json_durable(recipe_path, recipe)
        load_recipe(recipe_path)
        c['production_recipe'] = str(recipe_path)
    c['resident_data_directory'] = str(out / 'resident-data')
    c, sources = ns.continuation_config(c, stage, checkpoint=str(parent), out=str(out))
    c.pop('execution_transition_source_controller', None)
    c.pop('rolling_sft_resume', None)  # a same-slot retry control never carries into a new window
    if checkpoint_path is not None:  # the same bytes at another lineage's path (a relocated cloud parent)
        c['checkpoint'] = str(checkpoint_path)
    c['seed_trainer_parameters'] = copy.deepcopy(c['trainer_parameters'])
    if ident:
        c['id'] = ident
    c['branch_origin'] = {
        'parent': ref, 'parent_global_step': training['global_step'], 'actual_report': binding(report_path),
        'actual_completed_controller': binding(previous / 'state.json'), 'prior_control': binding(previous_config),
        'scored_graph_anchor': anchor, 'changed_controls': ({'refresh_after_updates': [prior_window, window]}
                                                            if window is not None and window != prior_window else {}),
        'reason': 'Next window of the same joint SFT continuation from the actual earned boundary.',
        'execution_admission': ({'folded_in_parent_history': folded[0]['qualification']['admission'],
                                 'controller_inheritance': 'omitted'} if folded else None),
        'new_optimizer_updates': 0, 'capability_lift_claimed': False}
    write_json_durable(config_out, c)
    result = {'prepared': str(config_out), 'out': str(out), 'parent': ref, 'global_step': training['global_step'],
              'source_roles': sorted(sources)}
    print(json.dumps(result), flush=True)
    return result


def adopt_prepared_window(config_out, out, previous_config, parent, ref):
    """The next window was prepared ahead of this boundary (an elastic DiLoCo session control relocated to this
    lineage, train/diloco/session.py): adopt it unchanged when it names exactly this boundary (the previous window's
    directory and the earned parent's sha256) and this window directory; anything else is refused, never rebuilt."""
    c = json.loads(config_out.read_text())
    marker = (c.get('branch_origin') or {}).get('prepared_next_window') or {}
    expected = {'prior_out': os.path.abspath(json.loads(Path(previous_config).read_text())['out']),
                'parent_sha256': ref['sha256']}
    if ({k: marker.get(k) for k in expected} != expected or os.path.abspath(c.get('out', '')) != os.path.abspath(out)
            or Path(c.get('checkpoint', '')).resolve() != parent or not c.get('mandatory_sft_only')):
        raise SystemExit(f'{config_out} exists and is not a control prepared for this exact boundary')
    result = {'prepared': str(config_out), 'out': str(out), 'parent': ref, 'adopted_prepared_window': marker}
    print(json.dumps(result), flush=True)
    return result


def rolling(config, config_out):
    """Bind an interrupted window's earned rolling slot (natlang.rolling-sft-resume/1); the control is otherwise the
    exact prior control."""
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    c = json.loads(Path(config).read_text())
    out = Path(c['out'])
    state = json.loads((out / 'state.json').read_text())
    if state.get('pending_native_checkpoint') or state.get('phase') not in ('mandatory_sft_pending', 'updating_mandatory_sft'):
        raise SystemExit(f'window is {state.get("phase")!r}: no interrupted slot to resume')
    slot = (out / 'learner/checkpoint.pt').resolve(strict=True)
    prev = Path(state['checkpoint']).resolve(strict=True)
    if slot == prev:
        raise SystemExit('rolling slot is the window parent: nothing earned to resume')
    training = CheckpointPolicy.load(slot)['training']
    if not training.get('optimizer') or not training.get('rng'):
        raise SystemExit('rolling slot lacks optimizer or RNG state')
    control = {'schema': 'natlang.rolling-sft-resume/1', 'checkpoint_path': str(slot),
               'checkpoint_sha256': sha256_file_hex_cached(slot), 'previous_checkpoint_path': str(prev),
               'previous_checkpoint_sha256': sha256_file_hex_cached(prev)}
    c = copy.deepcopy(c)
    c['rolling_sft_resume'] = control
    write_json_durable(_require_new(config_out), c)
    print(json.dumps({'prepared': str(config_out), 'global_step': training['global_step'], 'rolling': control}),
          flush=True)


# -- input closure ---------------------------------------------------------------------------------------------------

def _paths(value, out):
    if isinstance(value, str):
        if value.startswith('/') and '\n' not in value and len(value) < 4096:
            out.add(value)
    elif isinstance(value, list):
        for item in value:
            _paths(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _paths(item, out)


def admission_roots(config):
    """JSON files the control names under execution-transition/admission keys (top level and trainer parameters)."""
    roots = set()
    for scope in (config, config.get('trainer_parameters') or {}, config.get('seed_trainer_parameters') or {}):
        for key, value in scope.items():
            # task_selection_history: its record binds the source-controller snapshot rl.selection_history.initial
            # reads at controller start (2026-11-03 cloud interleaved-control start failure).
            if 'execution' in key or 'admission' in key or key == 'task_selection_history':
                found = set()
                _paths(value, found)
                roots |= {p for p in found if p.endswith('.json')}
    return sorted(roots)


def _bound_json_refs(value, out, suffix='.json'):
    """Paths of bound references ({"path": ..., "sha256": ...}) to ``suffix`` files inside a JSON value (a set of
    paths, or a dict path -> sha256)."""
    if isinstance(value, dict):
        path = value.get('path')
        if isinstance(path, str) and path.startswith('/') and path.endswith(suffix) and isinstance(value.get('sha256'), str):
            if isinstance(out, dict):
                out[path] = value['sha256']
            else:
                out.add(path)
        for item in value.values():
            _bound_json_refs(item, out, suffix)
    elif isinstance(value, list):
        for item in value:
            _bound_json_refs(item, out, suffix)


def admission_payload_references(roots, records):
    """Whole output directories of the sealed collections the admission records replay (the execution-transition
    qualification re-reads their rollouts, blocks and row results; 2026-11-03 cloud start failure on
    ffn-native67-neuralese-diagnostic-1102/blocks). Other non-JSON paths in those records are historical source
    payloads (evidence, not inputs) and are not shipped."""
    found = set()
    for current in [*roots, *records]:
        path = Path(current)
        if not path.is_file() or path.stat().st_size > NESTED_JSON_BYTES:
            continue
        try:
            value = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if isinstance(value, dict) and value.get('schema') == 'natlang.sealed-collection-continuation/1':
            out = value.get('out')
            if isinstance(out, str) and out.startswith('/') and Path(out).is_dir():
                found.add(out)
    return sorted(found)


def nested_checkpoint_references(roots, records):
    """Historical full checkpoints bound inside the admission records (execution-transition verification re-hashes the
    exact parent/child bytes at controller start): {path: sha256}. Shipped as exact checkpoint blobs, not files."""
    found = {}
    for current in [*roots, *records]:
        path = Path(current)
        if not path.is_file() or path.stat().st_size > NESTED_JSON_BYTES:
            continue
        try:
            _bound_json_refs(json.loads(path.read_text()), found, '.pt')
        except (ValueError, OSError):
            continue
    return dict(sorted(found.items()))


def _find_key(value, key, out):
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key and isinstance(item, str):
                out.add(item)
            else:
                _find_key(item, key, out)
    elif isinstance(value, list):
        for item in value:
            _find_key(item, key, out)


def protection_root_inputs(roots, records):
    """Files ``ProtectedWorlds.load`` reads for every ``protection_root`` the admission records name (the
    execution-transition qualification reloads the development packet's protection set at controller start;
    2026-11-03 namespace replay: the held packet cases under the code checkout's untracked runs/). Files tracked in
    a git checkout root arrive with the code bundle and are not shipped over it."""
    import subprocess
    from natlang_neuralese.common.custody import REGISTRY, _protected_source_paths, _read_protected_registry
    named = set()
    for current in [*roots, *records]:
        path = Path(current)
        if not path.is_file() or path.stat().st_size > NESTED_JSON_BYTES:
            continue
        try:
            _find_key(json.loads(path.read_text()), 'protection_root', named)
        except (ValueError, OSError):
            continue
    found = set()
    for root in sorted(named):
        root = Path(root)
        if not root.is_dir():
            continue
        tracked = set()
        if (root / '.git').exists():
            listing = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], capture_output=True, check=True).stdout
            tracked = {str(root / item) for item in listing.decode().split('\0') if item}
        registry = root / REGISTRY
        # The lexical path of every identity ProtectedWorlds.load(root) checks (the frozen runtime's own enumeration).
        for item in [str(registry), *(identity[0] for identity in
                                      _protected_source_paths(_read_protected_registry(registry), root))]:
            if item not in tracked:
                found.add(item)
    return sorted(found)


def nested_json_references(roots, *, depth=NESTED_JSON_DEPTH):
    """Small JSON records bound (path + sha256) inside control-named JSON files, transitively to ``depth``. Source
    manifests (native-graph-sources) are not expanded: their typed resolvers already name what training reads, and their
    per-shard receipts are provenance."""
    found, frontier = set(), list(dict.fromkeys(roots))
    for _ in range(depth):
        following = []
        for current in frontier:
            path = Path(current)
            if not path.is_file() or path.stat().st_size > NESTED_JSON_BYTES:
                continue
            try:
                value = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            if isinstance(value, dict) and str(value.get('schema', '')).startswith('natlang.native-graph-sources'):
                continue
            refs = set()
            _bound_json_refs(value, refs)
            for item in sorted(refs - found - set(roots)):
                if Path(item).is_file() and Path(item).stat().st_size <= NESTED_JSON_SHIP_BYTES:
                    found.add(item)
                    following.append(item)
        frontier = following
    return sorted(found)


TRACE_CALL = re.compile(r'^(?:\d+\s+)?(open|openat|stat|lstat|newfstatat|statx|access|faccessat2?)\('
                        r'(?:AT_FDCWD, |-?\d+, )?"([^"]+)".*\)\s+=\s+(-?\d+)')
# Host system trees (libraries, interpreters, devices, temporaries) are the sandbox's own, never closure inputs; on a
# merged-/usr host the loader opens /lib/<arch>/... (2026-11-03: shipped DGX arm64 libraries broke an x86 fetch).
TRACE_SYSTEM = tuple(f'/{name}/' for name in ('usr', 'etc', 'proc', 'sys', 'dev', 'tmp', 'var', 'run', 'lib', 'lib32',
                                              'lib64', 'libx32', 'bin', 'sbin', 'opt', 'snap', 'boot', 'root'))


def traced_reads(trace):
    """Regular files a traced controller start successfully opened or stat-ed (``strace -f -e trace=%file`` text):
    the inputs it actually reads, including those named only inside predecessor source manifests and verification
    records the static policy does not follow (2026-11-03 namespace replay). System paths and temporary files are
    not inputs."""
    found = set()
    with open(trace, errors='replace') as stream:
        for line in stream:
            match = TRACE_CALL.match(line)
            if not match or int(match.group(3)) < 0:
                continue
            path = match.group(2)
            if (path.startswith('/') and not path.startswith(TRACE_SYSTEM) and '/__pycache__/' not in path
                    and not os.path.realpath(path).startswith(TRACE_SYSTEM)):
                found.add(path)
    return sorted(path for path in found if Path(path).is_file())


def closure(config_path, *, max_file_bytes=4 * GIB, observed=None):
    """The files a window's controller and learner read, by the runtime's own resolvers: every absolute path the
    control names, the recipe resolver's files and stage inputs (manifests, shards, CE/focus/correction dependencies),
    and the runtime's protected-custody closure, plus the small JSON records the control's execution-transition/admission files bind
    (path + sha256) transitively to depth 4: controller start verifies admissions through them (e.g. the selected
    publication a source-controller admission names). Non-JSON paths recorded
    inside those files (historical source payloads, raw archives) are not followed: they are evidence, not inputs.
    Excluded and reported: the runtime itself (shipped as one archive), the base model (Hugging Face),
    the Python environment, checkpoints (shipped as exact blobs), the window's own output, scratch directories, and
    files above ``max_file_bytes`` (external provenance such as raw archives; never training payloads)."""
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.train.recipe import recipe_file_dependencies, _binding_files
    from natlang_neuralese.rl.native_raw_handoff import _recipe_plan
    config = json.loads(Path(config_path).read_text())
    runtime = Path((config.get('actor_environment') or {}).get('NATLANG_REPO') or Path(config['runtime']).parent)
    params = config.get('trainer_parameters') or {}
    roots = {
        'runtime': runtime, 'base': Path(config['base']), 'environment': Path(sys.prefix),
        'window-output': Path(config['out']),
    }
    for key in ('checkpoint_input_offload_directory',):
        if params.get(key):
            roots['scratch'] = Path(params[key])
    if config.get('llama_cpp_converter'):
        roots['llama-cpp'] = Path(config['llama_cpp_converter'])
    candidates, reasons = {}, {}

    def want(path, why):
        candidates.setdefault(path, set()).add(why)

    declared = set()
    _paths(config, declared)
    for path in declared:
        want(path, 'control')
    for item in recipe_file_dependencies(config['production_recipe']):
        want(item['path'] if isinstance(item, dict) else str(item), 'recipe')
    plan = _recipe_plan(Path(config['production_recipe']), 'cuda')
    for stage, roles in plan['stage_inputs'].items():
        for value in roles.values():
            if isinstance(value, dict) and 'path' in value:
                for item in _binding_files(value):
                    want(item['path'], f'stage-input:{stage}')
    # Execution-admission and custody records are verified through their nested historical file references at
    # controller start (native_execution_transition.verify_selected_publication reads the selected publication a
    # source-controller admission names, 2026-11-03 cloud start failure). Follow JSON references transitively from
    # the JSON files the control names: bounded to existing small .json files, so historical source payloads stay
    # evidence, not inputs.
    admission = admission_roots(config)
    nested = nested_json_references(admission)
    for item in nested:
        want(item, 'nested-admission-reference')
    historical_checkpoints = nested_checkpoint_references(admission, nested)
    for item in admission_payload_references(admission, nested):
        want(item, 'nested-admission-payload')
    for item in protection_root_inputs(admission, nested):
        want(item, 'protection-root')
    if observed:
        tracked = set()
        if (REPO / '.git').exists():  # the code bundle carries the checkout's tracked files
            import subprocess
            listing = subprocess.run(['git', '-C', str(REPO), 'ls-files', '-z'], capture_output=True, check=True).stdout
            tracked = {str(REPO / item) for item in listing.decode().split('\0') if item}
        for item in traced_reads(observed):
            if item not in tracked:
                want(item, 'observed-read')
    custody = runtime / 'runtime-input-closure.json'
    if custody.is_file():
        for item in json.loads(custody.read_text()).get('bound_files', []):
            want(item['path'], 'runtime-protected-custody')
    files, excluded, missing = {}, [], []
    for raw, why in sorted(candidates.items()):
        path, why = Path(raw), sorted(why)
        root = next((name for name, r in roots.items() if path == r or r in path.parents), None)
        if root is not None:
            excluded.append({'path': raw, 'reason': root, 'from': why})
            continue
        if path.name.endswith('.pt'):
            excluded.append({'path': raw, 'reason': 'checkpoint', 'from': why})
            continue
        if not path.exists():
            missing.append({'path': raw, 'from': why})
            continue
        members = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else []
        for member in members:
            key = str(member)
            if key in files:
                files[key]['from'] = sorted(set(files[key]['from']) | set(why))
                continue
            size = member.stat().st_size
            if size > max_file_bytes:
                excluded.append({'path': key, 'reason': 'large', 'bytes': size, 'from': why})
                continue
            files[key] = {'path': key, 'bytes': size, 'sha256': sha256_file_hex_cached(member), 'from': why}
    unique = {v['sha256'] for v in files.values()}
    return {'schema': 'natlang.sft-window-input-closure/1', 'config': str(Path(config_path).resolve()),
            'runtime': str(runtime), 'files': sorted(files.values(), key=lambda v: v['path']),
            'bytes': sum(v['bytes'] for v in files.values()), 'unique_contents': len(unique),
            'excluded': sorted(excluded, key=lambda v: v['path']), 'missing': sorted(missing, key=lambda v: v['path']),
            'historical_checkpoints': [{'path': p, 'sha256': d, 'present': Path(p).is_file()}
                                       for p, d in historical_checkpoints.items()]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('next')
    p.add_argument('--previous-config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--config-out', required=True)
    p.add_argument('--checkpoint-path')
    p.add_argument('--window', type=int)
    p.add_argument('--id')
    p = sub.add_parser('rolling')
    p.add_argument('--config', required=True)
    p.add_argument('--config-out', required=True)
    p = sub.add_parser('closure')
    p.add_argument('--config', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--observed-reads', help='strace -f file-call trace of this control\'s CPU controller start: '
                   'every regular file it read is a closure member')
    args = parser.parse_args(argv)
    if args.command == 'next':
        next_window(args.previous_config, args.out, args.config_out, checkpoint_path=args.checkpoint_path,
                    window=args.window, ident=args.id)
    elif args.command == 'rolling':
        rolling(args.config, args.config_out)
    else:
        from natlang_neuralese.common.jsonio import write_json_durable
        value = closure(args.config, observed=args.observed_reads)
        write_json_durable(Path(args.output), value)
        print(json.dumps({'files': len(value['files']), 'bytes': value['bytes'], 'excluded': len(value['excluded']),
                          'missing': len(value['missing'])}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
