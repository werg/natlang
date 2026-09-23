#!/usr/bin/env python3
"""Create the single production curriculum recipe; does not launch training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_pipeline import atomic_json


def recipe(repo, model="LiquidAI/LFM2.5-350M", revision=None, image=None, python="python", sources=None,
           source_limit=25000, synthetic=1000, teacher_programs=1000,
           teacher_model="Ternary-Bonsai-2-27B", teacher_server="http://127.0.0.1:8081", token_file=None, train_args=(), init_adapter=None, min_free_vram_mib=2048, inventories_override=None, captures_override=None, verified_turns_override=None, workspace_cases=()):
    repo = Path(repo).resolve()
    sources = sources or ["codesearchnet", "magicoder", "mceval", "tiny-codes", "xlam"]
    if "--full" in train_args:
        raise ValueError("This recipe chains LoRA adapters; full-weight stage recipes must chain checkpoint model directories explicitly")
    if any(arg.split('=')[0] in ('--model', '--model-revision') for arg in train_args):
        raise ValueError('select the student with recipe --model/--revision so render, audit and trainer agree')
    budget_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    budget_parser.add_argument('--max-len', type=int, default=8192)
    budget, _ = budget_parser.parse_known_args(list(train_args))
    if budget.max_len < 2:
        raise ValueError('training context budget must be at least two tokens')
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
                row.get('task', {}).get('program_ir', {}).get('semantics', {}).get('root', {}).get('$lambda', {}).get('codebase')
                and row.get('task', {}).get('program_ir', {}).get('source_layout')) for row in trajectories)
            if not converted:
                excluded_unit_captures.append({'path': str(manifest_path.parent), 'reason': 'local subfunctions were inlined or imported'})
                continue
        new_unit_turns.append(turns)
    verified_turns = ([str(Path(path).resolve()) for path in verified_turns_override] if verified_turns_override is not None else
                      [str(path) for path in (*proven_pilots, *new_unit_turns) if path.exists()])
    add("observe-source", ["node", f"{p}/ts-host/scripts/code-corpus/source-cases.mjs", "--output", f"{r}/source-observations.jsonl", "--execute", "--limit", "5000",
                           *[arg for path in observation_inputs for arg in ("--input", path)],
                           *[arg for path in captures for arg in ('--captures', path)]],
        [f"{p}/ts-host/scripts/code-corpus/source-cases.mjs", *observation_inputs, *captures],
        [f"{r}/source-observations.jsonl", f"{r}/source-observations.jsonl.report.json"])
    add("synthetic", ["node", f"{p}/ts-host/scripts/code-corpus/curriculum.mjs", "--out", f"{r}/synthetic", "--input", f"{r}/source-observations.jsonl", "--synthetic", str(synthetic), "--seed", "42", "--replay-synthetic", "--replay-candidates"],
        [f"{p}/ts-host/scripts/code-corpus/curriculum.mjs", f"{p}/ts-host/scripts/code-corpus/replay.mjs", f"{p}/ts-host/dist/native/runtime.js", f"{r}/source-observations.jsonl"],
        [f"{r}/synthetic/manifest.json", f"{r}/synthetic/verified-turns.jsonl", f"{r}/synthetic/teacher-programs.jsonl", f"{r}/synthetic/code-proposals.jsonl",
         f'{r}/synthetic/projection-rejected.jsonl', f'{r}/synthetic/replay-errors.jsonl'])
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
                                *[arg for ledger in ledgers for arg in ('--rejection-ledger', ledger)]]),
            [f'{p}/scripts/audit_training_corpus.py', f'{p}/scripts/render_training_corpus.py',
             f'{r}/{name}.sft.jsonl', f'{r}/{name}.sft.jsonl.manifest.json', *ledgers],
            [f'{r}/{name}.ready.jsonl', f'{r}/{name}.ready.jsonl.manifest.json',
             f'{r}/{name}.ready.jsonl.audit.json', f'{r}/{name}.ready.jsonl.rejected.jsonl'])
    def train(name, previous, lr):
        checkpoint = f"{r}/train-{name}/checkpoint/state.json"
        inputs = [f"{r}/{name}.ready.jsonl", f"{r}/{name}.ready.jsonl.manifest.json", f"{p}/scripts/train_lora.py", f"{p}/natlang/corpus.py",
                  f'{r}/training-readiness.json']
        args = [f"{p}/scripts/train_lora.py", f"{r}/{name}.ready.jsonl", f"{r}/train-{name}", *train_model_args, '--require-audit',
                "--epochs", "1", "--lr", lr, "--rank", "32", "--accum", "16", "--microbatch", "1", "--batch-tokens", "8192", "--max-len", "8192",
                "--save-every", "10", "--data-order", "source", "--skip-heldout-loss", "--no-merge", "--token-cache", f"{r}/{name}.tokens.sqlite"]
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
    add("teacher", ["node", f"{p}/ts-host/scripts/teacher-collector.mjs", f"{r}/teacher-programs.jsonl", f"{r}/teacher-jobs", f"{r}/teacher-trajectories.jsonl",
                    "--model-id", teacher_model, "--server", teacher_server, "--root-seed", "42", "--limit", str(teacher_programs), "--workers", "1"],
        [f"{r}/teacher-programs.jsonl", f"{p}/ts-host/dist/teacher/collector.js", f"{p}/ts-host/dist/native/runtime.js"],
        [f"{r}/teacher-trajectories.jsonl", f"{r}/teacher-trajectories.jsonl.manifest.json"])
    add("materialize-teacher", ["node", f"{p}/ts-host/scripts/materialize-native-teacher.mjs", f"{r}/teacher-trajectories.jsonl", f"{r}/teacher-turns.jsonl", "--replace"],
        [f"{r}/teacher-trajectories.jsonl", f"{p}/ts-host/dist/teacher/native-materializer.js"], [f"{r}/teacher-turns.jsonl"])
    add("prepare-teacher", py([f"{p}/scripts/prepare_training_stages.py", "--output", f"{r}/prepared-teacher", "--teacher", f"{r}/teacher-turns.jsonl", "--registry", f"{r}/prepared/splits.json"]),
        [f"{p}/scripts/prepare_training_stages.py", f"{r}/teacher-turns.jsonl", f"{r}/prepared/splits.json"],
        [f"{r}/prepared-teacher/manifest.json", f"{r}/prepared-teacher/teacher.jsonl"])
    render("teacher", f"{r}/prepared-teacher/teacher.jsonl")
    train("teacher", "coding", "0.00002")
    # All source observation/replay/teacher work uses one frozen interpreter build.
    frozen_stages = {"observe-source", "synthetic", "teacher-seeds", "teacher", "materialize-teacher"}
    for stage in stages:
        if stage["id"] in frozen_stages or stage['id'].startswith('capture-unit-test-'):
            for key in ("command", "inputs"):
                stage[key] = [value.replace(f"{p}/ts-host/", f"{r}/runtime-host/") for value in stage[key]]
            stage["inputs"].append(f"{r}/runtime-host/frozen-runtime.json")
    return {"version": "natlang.training_pipeline/1", "repository": str(repo), "stages": stages,
            "unit_test_corpus": {"included": [str(path) for path in new_unit_turns], "excluded": excluded_unit_captures}}


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
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--inventory", action="append", type=Path, help="explicit existing repository task file (repeatable); defaults to collected repository snapshots")
    parser.add_argument('--captures', action='append', type=Path, help='captured upstream-test calls (repeatable); defaults to existing repository capture snapshots')
    parser.add_argument('--verified-turns', action='append', type=Path, help='execution-verified native turns from unit-test replay (repeatable); defaults to existing repository pilot snapshots')
    parser.add_argument('--workspace-case', action='append', type=Path, default=[], help='JSON capture specification with workspace, source, test, function/functions and optional instruction/license; repeatable')
    parser.add_argument("--min-free-vram-mib", type=int, default=2048, help="GPU 0 availability gate; raise this for larger models")
    parser.add_argument("--train-arg", action="append", default=[], help="repeat as --train-arg=--load-in-4bit or --train-arg=VALUE to pass trainer options")
    args = parser.parse_args()
    config = recipe(Path(__file__).resolve().parents[1], args.model, args.revision, args.image, args.python,
                    source_limit=args.source_limit, synthetic=args.synthetic, teacher_programs=args.teacher_programs,
                    teacher_model=args.teacher_model, teacher_server=args.teacher_server, token_file=args.token_file,
                    train_args=args.train_arg, init_adapter=args.init_adapter, min_free_vram_mib=args.min_free_vram_mib,
                    inventories_override=args.inventory, captures_override=args.captures, verified_turns_override=args.verified_turns,
                    workspace_cases=args.workspace_case)
    if args.output.exists():
        if json.loads(args.output.read_text()) != config:
            raise ValueError("refusing to replace a different recipe")
    else:
        atomic_json(args.output, config)
    print(args.output)


if __name__ == "__main__":
    main()
