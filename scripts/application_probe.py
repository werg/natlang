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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:8080')
    ap.add_argument('--seed', type=int, default=19)
    ap.add_argument('--n', type=int, default=2, help='programs per family')
    ap.add_argument('--seconds', type=float, default=120, help='wall-clock budget per program, including callees')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    decoder = NativeCallDecoder(args.server, timeout=120)
    prompt = (ROOT / 'natlang/prompts/tools_delegate.md').read_text()
    rows = []
    for family, make in ARCHITECTURES.items():
        for index in range(args.n):
            seed = args.seed + index
            prog = make(random.Random(seed))
            logs = []

            def factory(lam):
                entry = {'function': lam.fn_name, 'body': lam.body, 'actions': [], 'transcript': []}
                logs.append(entry)
                return ToolAgent(decoder, system_prompt=prompt, temperature=0,
                                 log=entry['actions'], transcript=entry['transcript'], max_turns=24)

            rt = Runtime(factory, capabilities=prog.capabilities, max_episodes=48)
            start = time.monotonic()
            rt.deadline = start + args.seconds
            outcome, value = rt.run_root(prog.loader())
            actual = dump(value) if outcome.kind == 'done' else None
            correct = outcome.kind == 'done' and (prog.expected(actual) if callable(prog.expected) else actual == prog.expected)
            rows.append({'family': family, 'seed': seed, 'status': outcome.kind, 'correct': correct,
                         'value': actual, 'detail': outcome.detail, 'episodes': rt.episodes_started,
                         'seconds': time.monotonic() - start, 'logs': logs})
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({'server': args.server, 'rows': rows, 'usage': decoder.usage}, indent=2) + '\n')
            print(f'{family}/{seed}: {outcome.kind} correct={correct} episodes={rt.episodes_started} '
                  f'{rows[-1]["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
