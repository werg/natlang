#!/usr/bin/env python3
"""Run the model, not reference plans, on the architectural application generators."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from natlang.gen.architectures import ARCHITECTURES
from natlang.native import NativeCallDecoder
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.values import dump
from scripts.probe_parallel import completed_cases, total_usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:8080')
    ap.add_argument('--seed', type=int, default=19)
    ap.add_argument('--n', type=int, default=2, help='programs per family')
    ap.add_argument('--seconds', type=float, default=120, help='wall-clock budget per program, including callees')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--validation-feedback', choices=('local', 'caller'), default='caller')
    ap.add_argument("--workers", type=int, default=4, help="independent cases; match server slots")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("workers must be positive")
    prompt = (ROOT / 'natlang/prompts/tools_delegate.md').read_text()
    rows = []
    started = time.monotonic()
    jobs = [(family, args.seed + index) for family in ARCHITECTURES for index in range(args.n)]

    def run(job):
        family, seed = job
        make = ARCHITECTURES[family]
        decoder = NativeCallDecoder(args.server, timeout=120)
        prog = make(random.Random(seed))
        logs = []

        def factory(lam):
            entry = {'function': lam.fn_name, 'body': lam.body, 'actions': [], 'transcript': []}
            logs.append(entry)
            return ToolAgent(decoder, system_prompt=prompt, temperature=0,
                             validation_feedback=args.validation_feedback,
                             log=entry['actions'], transcript=entry['transcript'], max_turns=24)

        rt = Runtime(factory, capabilities=prog.capabilities, max_episodes=48)
        start = time.monotonic()
        rt.deadline = start + args.seconds
        outcome, value = rt.run_root(prog.loader())
        actual = dump(value) if outcome.kind == 'done' else None
        correct = outcome.kind == 'done' and (prog.expected(actual) if callable(prog.expected) else actual == prog.expected)
        return {'family': family, 'seed': seed, 'status': outcome.kind, 'correct': correct,
                'value': actual, 'detail': outcome.detail, 'episodes': rt.episodes_started,
                'seconds': time.monotonic() - start, 'logs': logs, 'usage': decoder.usage}

    for index, row in completed_cases(run, jobs, args.workers):
        row['case_index'] = index
        rows.append(row)
        rows.sort(key=lambda r: r['case_index'])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'server': args.server, 'validation_feedback': args.validation_feedback,
                                       'workers': args.workers, 'wall_seconds': time.monotonic() - started,
                                       'rows': rows, 'usage': total_usage(rows)}, indent=2) + '\n')
        print(f'{row["family"]}/{row["seed"]}: {row["status"]} correct={row["correct"]} '
              f'episodes={row["episodes"]} {row["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
