#!/usr/bin/env python3
"""Create the single production curriculum recipe; does not launch training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_pipeline import atomic_json


def recipe(repo, model="LiquidAI/LFM2.5-350M", revision=None, image=None, python="python", sources=None,
           source_limit=25000, synthetic=1000, teacher_programs=1000,
           teacher_model="Ternary-Bonsai-2-27B", teacher_server="http://127.0.0.1:8081", token_file=None, train_args=(), init_adapter=None, min_free_vram_mib=2048, inventories_override=None, captures_override=None):
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
        [f"{p}/scripts/freeze_training_runtime.py"], [f"{r}/runtime-host/frozen-runtime.json", f"{r}/runtime-host/dist", f"{r}/runtime-host/scripts", f"{r}/runtime-host/prelude.js"])
    typed_names = ("exercism-typescript.tasks.jsonl", "radashi.tasks.jsonl", "deno-std.final.tasks.jsonl", "remeda.tasks.jsonl")
    observation_inputs = [path for name in typed_names for path in inventories if Path(path).name == name]
    if inventories_override is not None:
        observation_inputs = inventories
    captures = ([str(Path(path).resolve()) for path in captures_override] if captures_override is not None else
                sorted(str(path) for path in (repo / 'data/direct-code-2026-09-23').glob('*/captures.jsonl')))
    add("observe-source", ["node", f"{p}/ts-host/scripts/code-corpus/source-cases.mjs", "--output", f"{r}/source-observations.jsonl", "--execute", "--limit", "5000",
                           *[arg for path in observation_inputs for arg in ("--input", path)],
                           *[arg for path in captures for arg in ('--captures', path)]],
        [f"{p}/ts-host/scripts/code-corpus/source-cases.mjs", *observation_inputs, *captures],
        [f"{r}/source-observations.jsonl", f"{r}/source-observations.jsonl.report.json"])
    add("synthetic", ["node", f"{p}/ts-host/scripts/code-corpus/curriculum.mjs", "--out", f"{r}/synthetic", "--input", f"{r}/source-observations.jsonl", "--synthetic", str(synthetic), "--seed", "42", "--replay-synthetic", "--replay-candidates"],
        [f"{p}/ts-host/scripts/code-corpus/curriculum.mjs", f"{p}/ts-host/scripts/code-corpus/replay.mjs", f"{p}/ts-host/dist/native/runtime.js", f"{r}/source-observations.jsonl"],
        [f"{r}/synthetic/manifest.json", f"{r}/synthetic/verified-turns.jsonl", f"{r}/synthetic/teacher-programs.jsonl", f"{r}/synthetic/code-proposals.jsonl",
         f'{r}/synthetic/projection-rejected.jsonl', f'{r}/synthetic/replay-errors.jsonl'])
    add("teacher-seeds", ["node", f"{p}/ts-host/scripts/code-corpus/teacher-seeds.mjs", f"{r}/synthetic/teacher-programs.jsonl", f"{r}/teacher-programs.jsonl", str(teacher_programs), "42"],
        [f"{p}/ts-host/scripts/code-corpus/teacher-seeds.mjs", f"{p}/ts-host/dist/teacher/synthetic-generator.js", f"{r}/synthetic/teacher-programs.jsonl"],
        [f"{r}/teacher-programs.jsonl", f"{r}/teacher-programs.jsonl.manifest.jsonl"])
    add("prepare", py([f"{p}/scripts/prepare_training_stages.py", "--output", f"{r}/prepared", "--code", f"{r}/bundle/train.jsonl", f"{r}/bundle/test.jsonl",
                       "--native", f"{r}/synthetic/verified-turns.jsonl", f"{r}/synthetic/code-proposals.jsonl", "--split-records", f"{r}/teacher-programs.jsonl"]),
        [f"{p}/scripts/prepare_training_stages.py", f"{r}/bundle/train.jsonl", f"{r}/bundle/test.jsonl", f"{r}/synthetic/verified-turns.jsonl", f"{r}/synthetic/code-proposals.jsonl", f"{r}/teacher-programs.jsonl"],
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
        if stage["id"] in frozen_stages:
            for key in ("command", "inputs"):
                stage[key] = [value.replace(f"{p}/ts-host/", f"{r}/runtime-host/") for value in stage[key]]
            stage["inputs"].append(f"{r}/runtime-host/frozen-runtime.json")
    return {"version": "natlang.training_pipeline/1", "repository": str(repo), "stages": stages}


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
    parser.add_argument("--min-free-vram-mib", type=int, default=2048, help="GPU 0 availability gate; raise this for larger models")
    parser.add_argument("--train-arg", action="append", default=[], help="repeat as --train-arg=--load-in-4bit or --train-arg=VALUE to pass trainer options")
    args = parser.parse_args()
    config = recipe(Path(__file__).resolve().parents[1], args.model, args.revision, args.image, args.python,
                    source_limit=args.source_limit, synthetic=args.synthetic, teacher_programs=args.teacher_programs,
                    teacher_model=args.teacher_model, teacher_server=args.teacher_server, token_file=args.token_file,
                    train_args=args.train_arg, init_adapter=args.init_adapter, min_free_vram_mib=args.min_free_vram_mib,
                    inventories_override=args.inventory, captures_override=args.captures)
    if args.output.exists():
        if json.loads(args.output.read_text()) != config:
            raise ValueError("refusing to replace a different recipe")
    else:
        atomic_json(args.output, config)
    print(args.output)


if __name__ == "__main__":
    main()
