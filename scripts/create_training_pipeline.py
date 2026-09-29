#!/usr/bin/env python3
"""Create the single production curriculum recipe; does not launch training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_pipeline import atomic_json


def file_project_has_subfunctions(ir):
    """Require the projected helper files, rather than the retired boxed lambda."""
    semantics = ir.get('semantics', {})
    layout = ir.get('source_layout', {})
    root = semantics.get('root')
    files = semantics.get('files', {})
    helpers = layout.get('subfunctions', {})
    return (ir.get('version') == 'natlang.program/2' and isinstance(root, str)
            and root == layout.get('root') and root in files and bool(helpers)
            and all(isinstance(path, str) and path in files for path in helpers.values()))


def current_program_turns(path):
    with Path(path).open() as stream:
        return all(json.loads(line).get('task', {}).get('program_ir', {}).get('version')
                   == 'natlang.program/2' for line in stream if line.strip())


def recipe(repo, model="LiquidAI/LFM2.5-350M", revision=None, image=None, python="python", sources=None,
           source_limit=25000, synthetic=1000, teacher_programs=1000,
           teacher_model="Ternary-Bonsai-2-27B", teacher_server="http://127.0.0.1:8081", teacher_provider=None,
           teacher_execution_plans=False, teacher_execution_plan_tokens=512,
           inline_shapes=2, token_file=None, train_args=(), init_adapter=None, min_free_vram_mib=2048, inventories_override=None, captures_override=None, verified_turns_override=None, workspace_cases=(), static_bundle=None, teacher_results_override=None):
    repo = Path(repo).resolve()
    sources = sources or ["codesearchnet", "magicoder", "mceval", "tiny-codes", "xlam"]
    if "--full" in train_args:
        raise ValueError("This recipe chains LoRA adapters; full-weight stage recipes must chain checkpoint model directories explicitly")
    if any(arg.split('=')[0] in ('--model', '--model-revision') for arg in train_args):
        raise ValueError('select the student with recipe --model/--revision so render, audit and trainer agree')
    budget_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    budget_parser.add_argument('--max-len', type=int, default=16384)
    budget, _ = budget_parser.parse_known_args(list(train_args))
    if budget.max_len < 2:
        raise ValueError('training context budget must be at least two tokens')
    if inline_shapes < 1:
        raise ValueError('inline shapes must be positive')
    if teacher_provider and teacher_server != "http://127.0.0.1:8081":
        raise ValueError('choose a teacher provider or server, not both')
    if not isinstance(teacher_execution_plan_tokens, int) or teacher_execution_plan_tokens < 1:
        raise ValueError('teacher execution plan tokens must be positive')
    track_registry = repo / 'ts-host/scripts/inline-curriculum/list-tracks.mjs'
    tracks = json.loads(subprocess.check_output(['node', str(track_registry)], cwd=repo, text=True))
    if not tracks or any(not track['generated_families'] for track in tracks):
        raise ValueError('each discovered inline track needs generated families')
    r = "${run}"
    p = "${repo}"
    def py(args, gpu=False):
        if not image:
            return [python, *args]
        return ["docker", "run", "--rm", "--stop-timeout", "1200", *(["--gpus", "all"] if gpu else []),
                "-v", f"{p}:{p}", "-v", f"{r}:{r}", "-w", p, "-e", f"HF_HOME={p}/models/hf", image, "python", *args]
    stages = []
    def add(name, command, inputs, outputs, **extra):
        stages.append({"id": name, "command": command, "inputs": inputs, "outputs": outputs, **extra})
    acquisition = ["node", f"{p}/ts-host/scripts/code-corpus/acquire-shards.mjs", "--out", f"{r}/sources",
                   "--sources", ",".join(sources), "--limit", str(source_limit)]
    if token_file:
        acquisition += ["--token-file", str(Path(token_file).resolve())]
    add("acquire", acquisition, [f"{p}/ts-host/scripts/code-corpus/acquire-shards.mjs", f"{p}/ts-host/scripts/code-corpus/datasets.mjs"], [f"{r}/sources/manifest.json"])
    # Keep existing repository inventories; acquisition obtains the HF sources afresh.
    inventories = sorted((repo / "data/direct-code-2026-09-23").glob("*.final.tasks.jsonl"))
    inventories += sorted((repo / "data/extended-code-2026-09-23").glob("*.tasks.jsonl"))
    inventories = [str(x) for x in inventories if x.name.split(".")[0] not in {"case2code", *sources}]
    if inventories_override is not None:
        inventories = [str(Path(x).resolve()) for x in inventories_override]
    add("assemble", ["node", f"{p}/ts-host/scripts/code-corpus/assemble-shards.mjs", f"{r}/bundle", f"{r}/sources", *inventories],
        [f"{r}/sources/manifest.json", f"{p}/ts-host/scripts/code-corpus/assemble.mjs", *inventories], [f"{r}/bundle/complete.jsonl", f"{r}/bundle/train.jsonl", f"{r}/bundle/test.jsonl", f'{r}/bundle/rejected.jsonl'])
    add("freeze-runtime", [python, f"{p}/scripts/freeze_training_runtime.py", f"{p}/ts-host", f"{r}/runtime-host"],
        [f"{p}/scripts/freeze_training_runtime.py"], [f"{r}/runtime-host/frozen-runtime.json", f"{r}/runtime-host/dist", f"{r}/runtime-host/scripts", f"{r}/runtime-host/src", f"{r}/runtime-host/prelude.js"])
    failure_cases = f"{r}/failure-repair-cases.jsonl"
    add("freeze-failure-corpus", ["node", f"{r}/runtime-host/scripts/failure-corpus/freeze.mjs", failure_cases],
        [f"{r}/runtime-host/frozen-runtime.json", f"{r}/runtime-host/scripts/failure-corpus/cases.mjs",
         f"{r}/runtime-host/scripts/failure-corpus/freeze.mjs"],
        [failure_cases, f"{failure_cases}.manifest.json"])
    typed_names = ("exercism-typescript.tasks.jsonl", "radashi.tasks.jsonl", "deno-std.final.tasks.jsonl", "remeda.tasks.jsonl")
    observation_inputs = [path for name in typed_names for path in inventories if Path(path).name == name]
    if inventories_override is not None:
        observation_inputs = inventories
    captures = ([str(Path(path).resolve()) for path in captures_override] if captures_override is not None else
                sorted(str(path) for path in (repo / 'data/direct-code-2026-09-23').glob('*/captures.jsonl')))
    proven_pilots = (
        repo / 'data/direct-code-2026-09-23/chunk-native-final.jsonl.turns.jsonl',
        repo / 'data/direct-code-2026-09-23/d3-pilot/native-replay.jsonl.turns.jsonl',
    )
    new_unit_turns = []
    excluded_unit_captures = []
    for manifest_path in sorted((repo / 'data/direct-code-2026-09-23/unit-test-corpus').glob('*/manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        turns = manifest_path.parent / 'native-replay.jsonl.turns.jsonl'
        replay = manifest.get('native_replay', {})
        if replay.get('accepted', 0) < 1 or not turns.is_file():
            continue
        if hashlib.sha256(turns.read_bytes()).hexdigest() != replay.get('turns_sha256'):
            raise ValueError(f'unit-test turns do not match capture manifest: {turns}')
        tasks_path = manifest_path.parent / 'tasks.jsonl'
        trajectories_path = manifest_path.parent / 'native-replay.jsonl'
        if not tasks_path.is_file() or not trajectories_path.is_file():
            excluded_unit_captures.append({'path': str(manifest_path.parent), 'reason': 'missing projection evidence'})
            continue
        tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
        if any(task.get('function', {}).get('recursive') for task in tasks):
            excluded_unit_captures.append({'path': str(manifest_path.parent), 'reason': 'recursive function graph'})
            continue
        needs_codebase = any(task.get('function', {}).get('helpers') or any(
            imp.get('specifier', '').startswith('.') for imp in task.get('function', {}).get('imports', [])) for task in tasks)
        if needs_codebase:
            trajectories = [json.loads(line) for line in trajectories_path.read_text().splitlines() if line.strip()]
            converted = trajectories and all(not row.get('outcome', {}).get('accepted') or (
                file_project_has_subfunctions(row.get('task', {}).get('program_ir', {}))) for row in trajectories)
            if not converted:
                excluded_unit_captures.append({'path': str(manifest_path.parent), 'reason': 'local subfunctions were inlined or imported'})
                continue
        new_unit_turns.append(turns)
    compatible_turns, excluded_turns, saved_replays = [], [], []
    for path in (*proven_pilots, *new_unit_turns):
        if not path.exists():
            continue
        if current_program_turns(path):
            compatible_turns.append(str(path))
        else:
            excluded_turns.append({'path': str(path), 'reason': 'retired program/prompt snapshot; replay source on current runtime'})
            tasks_path, captures_path = path.parent / 'tasks.jsonl', path.parent / 'captures.jsonl'
            if path in new_unit_turns and tasks_path.is_file() and captures_path.is_file():
                tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
                if tasks and all(not task.get('function', {}).get(key) for task in tasks
                                 for key in ('helpers', 'imports', 'recursive')):
                    saved_replays.append((tasks_path, captures_path, len(tasks)))
    verified_turns = ([str(Path(path).resolve()) for path in verified_turns_override] if verified_turns_override is not None else compatible_turns)
    for path in verified_turns:
        if Path(path).is_file() and not current_program_turns(path):
            raise ValueError(f'retired program/prompt snapshot must be replayed before training: {path}')
    default_static = [repo / 'data/teacher/source-backed/static.manifest.json',
                      repo / 'data/teacher/recovered/static.manifest.json',
                      repo / 'data/teacher/directory-expansion/static.manifest.json',
                      repo / 'data/teacher/workflowevals/static.manifest.json']
    static_manifests = ([] if static_bundle is False else
                        [path for path in default_static if path.exists()] if static_bundle is None else
                        [Path(path).resolve() for path in (static_bundle if isinstance(static_bundle, (list, tuple)) else [static_bundle])])
    if len(set(static_manifests)) != len(static_manifests):
        raise ValueError('duplicate static bundle manifests')
    included_statics = []
    for index, static_manifest in enumerate(static_manifests):
        suffix = '' if index == 0 else f'-{index}'
        manifest = json.loads(static_manifest.read_text())
        if manifest.get('version') != 'natlang.source_static_bundle/1' or not manifest.get('cases'):
            raise ValueError('invalid static source bundle')
        static_inputs = []
        for field in ('ir', 'results'):
            entry = manifest[field]
            path = (static_manifest.parent / entry['path']).resolve()
            if not path.is_relative_to(static_manifest.parent.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('static source bundle checksum/path mismatch')
            static_inputs.append(str(path))
        static_turns = f'{r}/static-source{suffix}.turns.jsonl'
        add(f'validate-static-sources{suffix}', ['node', f'{p}/ts-host/scripts/inline-curriculum/static-bundle-input.mjs',
                                      str(static_manifest), '--turns-out', static_turns],
            [str(static_manifest), *static_inputs, f'{p}/ts-host/scripts/inline-curriculum/static-bundle-input.mjs',
             f'{p}/ts-host/scripts/jsonl-stream.mjs',
             f'{p}/ts-host/dist/teacher/source-conversion.js', f'{p}/ts-host/dist/teacher/native-materializer.js'],
            [static_turns, f'{static_turns}.manifest.json'])
        verified_turns.append(static_turns)
        included_statics.append({'manifest': str(static_manifest), 'cases': manifest['cases'], 'model_calls': 0})
    included_static = included_statics[0] if included_statics else None
    add("observe-source", ["node", f"{p}/ts-host/scripts/code-corpus/source-cases.mjs", "--output", f"{r}/source-observations.jsonl", "--execute", "--limit", "5000",
                           *[arg for path in observation_inputs for arg in ("--input", path)],
                           *[arg for path in captures for arg in ('--captures', path)]],
        [f"{p}/ts-host/scripts/code-corpus/source-cases.mjs", *observation_inputs, *captures],
        [f"{r}/source-observations.jsonl", f"{r}/source-observations.jsonl.report.json"])
    add("synthetic", ["node", f"{p}/ts-host/scripts/code-corpus/curriculum.mjs", "--out", f"{r}/synthetic", "--input", f"{r}/source-observations.jsonl", "--synthetic", str(synthetic), "--seed", "42", "--replay-synthetic", "--replay-candidates"],
        [f"{p}/ts-host/scripts/code-corpus/curriculum.mjs", f"{p}/ts-host/scripts/code-corpus/replay.mjs", f"{p}/ts-host/dist/native/runtime.js", f"{r}/source-observations.jsonl"],
        [f"{r}/synthetic/manifest.json", f"{r}/synthetic/verified-turns.jsonl", f"{r}/synthetic/teacher-programs.jsonl", f"{r}/synthetic/code-proposals.jsonl",
         f'{r}/synthetic/projection-rejected.jsonl', f'{r}/synthetic/replay-errors.jsonl'])
    if verified_turns_override is None:
        for index, (tasks_path, captures_path, task_count) in enumerate(saved_replays):
            rows = f'{r}/saved-source-{index:04d}.jsonl'
            # Regenerate the tool decisions from captured inputs; never translate
            # obsolete prompt text or reinterpret captures as independent tests.
            add(f'replay-saved-source-{index:04d}', ['node', f'{p}/ts-host/scripts/code-corpus/replay.mjs',
                '--execute', '--input', str(tasks_path), '--captures', str(captures_path),
                '--limit', str(task_count), '--cases', '20', '--output', rows],
                [str(tasks_path), str(captures_path), f'{p}/ts-host/scripts/code-corpus/replay.mjs'],
                [rows, f'{rows}.turns.jsonl', f'{rows}.rejected.jsonl'])
            verified_turns.append(f'{rows}.turns.jsonl')
    for index, case_file in enumerate(workspace_cases):
        case_path = Path(case_file).resolve()
        case = json.loads(case_path.read_text())
        required = ('workspace', 'source', 'test')
        if not isinstance(case, dict) or any(not isinstance(case.get(key), str) or not case[key] for key in required):
            raise ValueError(f'unit-test case {case_path} needs nonempty workspace, source, test strings')
        functions = case.get('functions', [case.get('function')])
        if not isinstance(functions, list) or not functions or any(not isinstance(name, str) or not name for name in functions):
            raise ValueError(f'unit-test case {case_path} needs a function or nonempty functions list')
        workspace_setting = Path(case['workspace'])
        workspace = (workspace_setting if workspace_setting.is_absolute() else case_path.parent / workspace_setting).resolve()
        if not workspace.is_dir():
            raise ValueError(f'unit-test workspace does not exist: {workspace}')
        for key in ('source', 'test'):
            target = (workspace / case[key]).resolve()
            if not target.is_relative_to(workspace) or not target.is_file():
                raise ValueError(f'unit-test {key} must be an existing workspace-relative file: {case[key]}')
        case_out = f'{r}/captured-unit-tests/{index:04d}'
        command = ['node', f'{p}/ts-host/scripts/code-corpus/workspace-pilot.mjs', '--execute',
                   '--workspace', str(workspace), '--source', case['source'], '--test', case['test'],
                   *[arg for name in functions for arg in ('--function', name)], '--output', case_out]
        for option in ('license', 'instruction'):
            if case.get(option):
                command += [f'--{option}', case[option]]
        add(f'capture-unit-test-{index:04d}', command,
            [str(case_path), str(workspace / 'package.json'), str(workspace / case['source']),
             str(workspace / case['test']), *[str(workspace / name) for name in ('package-lock.json', 'npm-shrinkwrap.json', 'yarn.lock', 'pnpm-lock.yaml') if (workspace / name).exists()],
             f'{p}/ts-host/scripts/code-corpus/workspace-pilot.mjs', f'{p}/ts-host/scripts/code-corpus/replay.mjs'],
            [f'{case_out}/manifest.json', f'{case_out}/native-replay.jsonl.turns.jsonl'])
        verified_turns.append(f'{case_out}/native-replay.jsonl.turns.jsonl')
    add("teacher-seeds", ["node", f"{p}/ts-host/scripts/code-corpus/teacher-seeds.mjs", f"{r}/synthetic/teacher-programs.jsonl", f"{r}/teacher-programs.jsonl", str(teacher_programs), "42", failure_cases],
        [f"{p}/ts-host/scripts/code-corpus/teacher-seeds.mjs", f"{p}/ts-host/dist/teacher/synthetic-generator.js", f"{r}/synthetic/teacher-programs.jsonl", failure_cases],
        [f"{r}/teacher-programs.jsonl", f"{r}/teacher-programs.jsonl.manifest.jsonl"])
    add("prepare", py([f"{p}/scripts/prepare_training_stages.py", "--output", f"{r}/prepared", "--code", f"{r}/bundle/train.jsonl", f"{r}/bundle/test.jsonl",
                       "--native", f"{r}/synthetic/verified-turns.jsonl", f"{r}/synthetic/code-proposals.jsonl", *verified_turns,
                       "--split-records", f"{r}/teacher-programs.jsonl"]),
        [f"{p}/scripts/prepare_training_stages.py", f"{r}/bundle/train.jsonl", f"{r}/bundle/test.jsonl", f"{r}/synthetic/verified-turns.jsonl", f"{r}/synthetic/code-proposals.jsonl", *verified_turns, f"{r}/teacher-programs.jsonl"],
        [f"{r}/prepared/manifest.json", f"{r}/prepared/general.jsonl", f"{r}/prepared/coding.jsonl", f"{r}/prepared/splits.json"])
    model_args = ["--model", model] + (["--revision", revision] if revision else [])
    train_model_args = ["--model", model] + (["--model-revision", revision] if revision else [])
    add('training-readiness', py([f'{p}/scripts/training_readiness.py', '--output', f'{r}/training-readiness.json']),
        [f'{p}/scripts/training_readiness.py', f'{p}/scripts/train_lora.py'], [f'{r}/training-readiness.json'])
    def render(name, source):
        ledgers = [f'{r}/{name}.sft.jsonl.rejected.jsonl'] + ([f'{r}/bundle/rejected.jsonl'] if name == 'general' else
                   [f'{r}/synthetic/projection-rejected.jsonl', f'{r}/synthetic/replay-errors.jsonl'] if name == 'coding' else [])
        add(f"render-{name}", py([f"{p}/scripts/render_training_corpus.py", "--inputs", source, "--output", f"{r}/{name}.sft.jsonl", *model_args]),
            [f"{p}/scripts/render_training_corpus.py", source], [f"{r}/{name}.sft.jsonl", f"{r}/{name}.sft.jsonl.manifest.json", f'{r}/{name}.sft.jsonl.rejected.jsonl'])
        add(f'audit-{name}', py([f'{p}/scripts/audit_training_corpus.py', '--input', f'{r}/{name}.sft.jsonl',
                                '--output', f'{r}/{name}.ready.jsonl', '--max-len', str(budget.max_len), *model_args,
                                *([arg for track in ['seed-teacher', *(item['id'] for item in tracks)]
                                   for arg in ('--require-track', track)] if name == 'joint' else []),
                                *[arg for ledger in ledgers for arg in ('--rejection-ledger', ledger)]]),
            [f'{p}/scripts/audit_training_corpus.py', f'{p}/scripts/render_training_corpus.py',
             f'{r}/{name}.sft.jsonl', f'{r}/{name}.sft.jsonl.manifest.json', *ledgers],
            [f'{r}/{name}.ready.jsonl', f'{r}/{name}.ready.jsonl.manifest.json',
             f'{r}/{name}.ready.jsonl.audit.json', f'{r}/{name}.ready.jsonl.rejected.jsonl'])
    def train(name, previous, lr):
        checkpoint = f"{r}/train-{name}/checkpoint/state.json"
        inputs = [f"{r}/{name}.ready.jsonl", f"{r}/{name}.ready.jsonl.manifest.json", f"{p}/scripts/train_lora.py", f"{p}/scripts/corpus.py",
                  f'{r}/training-readiness.json']
        args = [f"{p}/scripts/train_lora.py", f"{r}/{name}.ready.jsonl", f"{r}/train-{name}", *train_model_args, '--require-audit',
                "--epochs", "1", "--lr", lr, "--rank", "32", "--accum", "16", "--microbatch", "1", "--batch-tokens", "16384", "--max-len", "16384",
                "--save-every", "10", "--data-order", "shuffle" if name == 'joint' else "source",
                "--skip-heldout-loss", "--no-merge", "--token-cache", f"{r}/{name}.tokens.sqlite"]
        if previous:
            args += ["--init-adapter", f"{r}/train-{previous}/checkpoint/weights"]
            inputs += [f"{r}/train-{previous}/checkpoint/state.json", f"{r}/train-{previous}/checkpoint/weights/adapter_model.safetensors"]
        elif init_adapter:
            args += ["--init-adapter", str(Path(init_adapter).resolve())]
            inputs += [str(Path(init_adapter).resolve() / "adapter_model.safetensors")]
        args += list(train_args)
        add(f"train-{name}", py(args, gpu=True), inputs,
            [checkpoint, f"{r}/train-{name}/checkpoint/weights/adapter_model.safetensors"], training_state=checkpoint,
            min_free_vram_mib=min_free_vram_mib)
    render("general", f"{r}/prepared/general.jsonl")
    train("general", None, "0.0001")
    render("coding", f"{r}/prepared/coding.jsonl")
    train("coding", "general", "0.00005")
    teacher_connection = ['--provider', teacher_provider] if teacher_provider else ['--server', teacher_server]
    teacher_planning = (['--execution-plans', '--execution-plan-tokens', str(teacher_execution_plan_tokens)]
                        if teacher_execution_plans else [])
    add("teacher", ["node", f"{p}/ts-host/scripts/teacher-collector.mjs", f"{r}/teacher-programs.jsonl", f"{r}/teacher-jobs", f"{r}/teacher-trajectories.jsonl",
                    "--model-id", teacher_model, *teacher_connection, *teacher_planning,
                    "--root-seed", "42", "--limit", str(teacher_programs), "--workers", "1"],
        [f"{r}/teacher-programs.jsonl", f"{p}/ts-host/dist/teacher/collector.js", f"{p}/ts-host/dist/native/runtime.js"],
        [f"{r}/teacher-trajectories.jsonl", f"{r}/teacher-trajectories.jsonl.manifest.json"])
    add("materialize-teacher", ["node", f"{p}/ts-host/scripts/materialize-native-teacher.mjs", f"{r}/teacher-trajectories.jsonl", f"{r}/teacher-turns.jsonl", "--replace"],
        [f"{r}/teacher-trajectories.jsonl", f"{p}/ts-host/dist/teacher/native-materializer.js"], [f"{r}/teacher-turns.jsonl"])
    teacher_tracks = [('seed-teacher', f'{r}/teacher-turns.jsonl')]
    for track in tracks:
        name = track['id']
        if not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f'invalid track id: {name}')
        base = f'{r}/tracks/{name}'
        pool = f'{base}/cases.ir.jsonl'
        trajectories = f'{base}/trajectories.jsonl'
        admitted = f'{base}/admitted.jsonl'
        turns = f'{base}/turns.jsonl'
        add(f'build-{name}', ['node', f'{p}/ts-host/scripts/inline-curriculum/build.mjs',
                             '--seed', '42', '--shapes', str(inline_shapes), '--track', name,
                             '--families', ','.join(track['generated_families']), '--out', pool],
            [f'{p}/ts-host/scripts/inline-curriculum/list-tracks.mjs',
             f'{p}/ts-host/scripts/inline-curriculum/build.mjs', f'{p}/ts-host/scripts/inline-curriculum/families.mjs',
             f'{r}/runtime-host/frozen-runtime.json'], [pool, f'{pool}.report.json'])
        add(f'collect-{name}', ['node', f'{p}/ts-host/scripts/teacher-collector.mjs', pool, f'{base}/jobs', trajectories,
                               '--model-id', teacher_model, *teacher_connection, *teacher_planning,
                               '--root-seed', '42', '--all', '--workers', '1'],
            [pool, f'{r}/runtime-host/frozen-runtime.json', f'{p}/ts-host/dist/teacher/collector.js'],
            [trajectories, f'{trajectories}.manifest.json'])
        add(f'admit-{name}', ['node', f'{p}/ts-host/scripts/inline-curriculum/admit.mjs', trajectories,
                             '--ledger', f'{base}/admission.jsonl', '--admitted', admitted],
            [trajectories, f'{p}/ts-host/scripts/inline-curriculum/admit.mjs', f'{r}/runtime-host/frozen-runtime.json'],
            [f'{base}/admission.jsonl', f'{base}/admission.coverage.json', admitted])
        add(f'materialize-{name}', ['node', f'{p}/ts-host/scripts/materialize-native-teacher.mjs', admitted, turns, '--replace'],
            [admitted, f'{p}/ts-host/dist/teacher/native-materializer.js', f'{r}/runtime-host/frozen-runtime.json'], [turns])
        teacher_tracks.append((name, turns))
    # Existing static references were part of the shell recipe but fell out of
    # the staged recipe. Re-admit and materialize them on the frozen build.
    default_results = [repo / 'runs/inline-curriculum' / name for name in
                       ('ref-v1.results.jsonl', 'ref-composed-v1.results.jsonl')]
    if teacher_results_override is None:
        # Snapshot completed jobs across historical and current campaigns, rather
        # than silently omitting everything outside the two reference exports.
        generated = subprocess.check_output(
            ['node', str(repo / 'ts-host/scripts/snapshot-generated-training.mjs'),
             '--repo', str(repo)], cwd=repo, text=True).strip()
        default_results.append(Path(generated))
    teacher_results = ([str(Path(path).resolve()) for path in teacher_results_override]
                       if teacher_results_override is not None else
                       [str(path) for path in default_results if path.is_file()])
    if len(set(teacher_results)) != len(teacher_results):
        raise ValueError('duplicate existing teacher results input')
    if teacher_results:
        existing = f'{r}/existing-curriculum'
        admitted, turns = f'{existing}.admitted.jsonl', f'{existing}.turns.jsonl'
        add('admit-existing-curriculum', ['node', f'{p}/ts-host/scripts/inline-curriculum/admit.mjs',
            *teacher_results, '--ledger', f'{existing}.admission.jsonl', '--admitted', admitted],
            [*teacher_results, f'{p}/ts-host/scripts/inline-curriculum/admit.mjs'],
            [admitted, f'{existing}.admission.jsonl', f'{existing}.admission.coverage.json'])
        add('materialize-existing-curriculum', ['node', f'{p}/ts-host/scripts/materialize-native-teacher.mjs',
            admitted, turns, '--replace'], [admitted, f'{p}/ts-host/dist/teacher/native-materializer.js'], [turns])
        teacher_tracks.append(('existing-curriculum', turns))
    turn_paths = [path for _, path in teacher_tracks]
    add("prepare-teacher", py([f"{p}/scripts/prepare_training_stages.py", "--output", f"{r}/prepared-teacher", "--teacher", *turn_paths, "--registry", f"{r}/prepared/splits.json"]),
        [f"{p}/scripts/prepare_training_stages.py", *turn_paths, f"{r}/prepared/splits.json"],
        [f"{r}/prepared-teacher/manifest.json", f"{r}/prepared-teacher/teacher.jsonl"])
    add('assemble-joint', py([f'{p}/scripts/assemble_joint_curriculum.py', '--output', f'{r}/joint.jsonl',
                              '--general', f'{r}/prepared/general.jsonl', '--coding', f'{r}/prepared/coding.jsonl',
                              '--teacher', f'{r}/prepared-teacher/teacher.jsonl',
                              *[arg for name, path in teacher_tracks for arg in ('--track', name, path)]]),
        [f'{p}/scripts/assemble_joint_curriculum.py', f'{r}/prepared/general.jsonl', f'{r}/prepared/coding.jsonl',
         f'{r}/prepared-teacher/teacher.jsonl', *turn_paths],
        [f'{r}/joint.jsonl', f'{r}/joint.jsonl.manifest.json'])
    render('joint', f'{r}/joint.jsonl')
    add('audit-joint-mix', py([f'{p}/scripts/audit_training_mix.py', '--input', f'{r}/joint.ready.jsonl',
                              '--output', f'{r}/joint.mix.json', '--reducer-share', '0.25', '--require-target']),
        [f'{p}/scripts/audit_training_mix.py', f'{r}/joint.ready.jsonl', f'{r}/joint.ready.jsonl.manifest.json'],
        [f'{r}/joint.mix.json'])
    train('joint', 'coding', '0.00002')
    stages[-1]['inputs'].append(f'{r}/joint.mix.json')
    # All source observation/replay/teacher work uses one frozen interpreter build.
    frozen_stages = {"observe-source", "synthetic", "teacher-seeds", "teacher", "materialize-teacher", 'validate-static-sources'}
    for stage in stages:
        if stage["id"] in frozen_stages or stage['id'].startswith(('validate-static-sources-', 'capture-unit-test-', 'replay-saved-source-', 'build-', 'collect-', 'admit-', 'materialize-')):
            for key in ("command", "inputs"):
                stage[key] = [value.replace(f"{p}/ts-host/", f"{r}/runtime-host/") for value in stage[key]]
            stage["inputs"].append(f"{r}/runtime-host/frozen-runtime.json")
    result = {"version": "natlang.training_pipeline/1", "repository": str(repo), "stages": stages,
            "training_tracks": tracks,
            'static_source_bundle': included_static,
            'static_source_bundles': included_statics,
            'existing_teacher_results': teacher_results,
            'excluded_legacy_turns': excluded_turns,
            "unit_test_corpus": {"included": [str(path) for path in new_unit_turns if str(path) in compatible_turns],
                                 "excluded": excluded_unit_captures,
                                 'replayed_sources': [str(tasks) for tasks, _, _ in saved_replays]
                                     if verified_turns_override is None else []}}
    result['data_inventory_explicit_input_override'] = (teacher_results_override is not None or static_bundle is not None)
    from inventory_training_data import catalog
    inventory_path, inventory = catalog(repo, result)
    result['data_inventory'] = {'report': str(inventory_path), 'by_status': inventory['by_status'],
                              'missing_required_default_inputs': inventory['missing_required_default_inputs'],
                              'included_quality_blockers': inventory['included_quality_blockers'],
                              'policy': inventory['policy']}
    ready = f'{r}/data-inventory.ready.json'
    guard = {'id': 'audit-data-inventory',
             'command': [python, f'{p}/scripts/inventory_training_data.py', '--check-report', str(inventory_path), '--ready-out', ready],
             'inputs': [str(inventory_path), f'{p}/scripts/inventory_training_data.py', str(repo / 'training/data_sources.json')],
             'outputs': [ready]}
    index = next(i for i, stage in enumerate(stages) if stage['id'] == 'train-joint')
    stages.insert(index, guard)
    stages[index + 1]['inputs'].append(ready)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--revision")
    parser.add_argument("--image", help="optional existing Docker training image")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-limit", type=int, default=25000)
    parser.add_argument("--synthetic", type=int, default=1000)
    parser.add_argument("--teacher-programs", type=int, default=1000)
    parser.add_argument("--teacher-model", default="Ternary-Bonsai-2-27B")
    parser.add_argument("--teacher-server", default="http://127.0.0.1:8081")
    parser.add_argument("--teacher-provider", help="Pi provider for teacher collection, using saved subscription/API credentials")
    parser.add_argument("--teacher-execution-plans", action="store_true",
                        help="elicit a required-tool execution plan before each teacher action")
    parser.add_argument("--teacher-execution-plan-tokens", type=int, default=512,
                        help="maximum tokens for each optional execution plan")
    parser.add_argument("--inline-shapes", type=int, default=2, help="generated cases per inline curriculum family")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--inventory", action="append", type=Path, help="explicit existing repository task file (repeatable); defaults to collected repository snapshots")
    parser.add_argument('--captures', action='append', type=Path, help='captured upstream-test calls (repeatable); defaults to existing repository capture snapshots')
    parser.add_argument('--verified-turns', action='append', type=Path, help='execution-verified native turns from unit-test replay (repeatable); defaults to existing repository pilot snapshots')
    parser.add_argument('--workspace-case', action='append', type=Path, default=[], help='JSON capture specification with workspace, source, test, function/functions and optional instruction/license; repeatable')
    parser.add_argument('--static-bundle', type=Path, action='append', help='validated static source manifest; repeatable; defaults to source-backed, recovered, directory-expansion and retired WorkflowEvals manifests when present')
    parser.add_argument('--no-static-bundle', action='store_true', help='omit the static source bundle from this recipe')
    parser.add_argument('--teacher-results', action='append', type=Path, help='existing curriculum result snapshots (repeatable); defaults to the two static reference sets')
    parser.add_argument('--no-existing-teacher-results', action='store_true', help='omit existing curriculum snapshots')
    parser.add_argument("--min-free-vram-mib", type=int, default=2048, help="GPU 0 availability gate; raise this for larger models")
    parser.add_argument("--train-arg", action="append", default=[], help="repeat as --train-arg=--load-in-4bit or --train-arg=VALUE to pass trainer options")
    args = parser.parse_args()
    config = recipe(Path(__file__).resolve().parents[1], args.model, args.revision, args.image, args.python,
                    source_limit=args.source_limit, synthetic=args.synthetic, teacher_programs=args.teacher_programs,
                    teacher_model=args.teacher_model, teacher_server=args.teacher_server, teacher_provider=args.teacher_provider,
                    teacher_execution_plans=args.teacher_execution_plans,
                    teacher_execution_plan_tokens=args.teacher_execution_plan_tokens,
                    inline_shapes=args.inline_shapes, token_file=args.token_file,
                    train_args=args.train_arg, init_adapter=args.init_adapter, min_free_vram_mib=args.min_free_vram_mib,
                    inventories_override=args.inventory, captures_override=args.captures, verified_turns_override=args.verified_turns,
                    workspace_cases=args.workspace_case, static_bundle=False if args.no_static_bundle else args.static_bundle,
                    teacher_results_override=[] if args.no_existing_teacher_results else args.teacher_results)
    if args.output.exists():
        if json.loads(args.output.read_text()) != config:
            raise ValueError("refusing to replace a different recipe")
    else:
        atomic_json(args.output, config)
    print(args.output)


if __name__ == "__main__":
    main()
