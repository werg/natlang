#!/usr/bin/env python3
"""Fill the LFM3 sandbox templates in this directory from one values file, so a launch needs no hand editing.

    render.py VALUES.json OUT_DIR        writes every *.json template, filled, to OUT_DIR, then validates the jobs

VALUES.json:
    {"placeholders": {"RUN_ID": "lfm3-sft-diloco-20261029", "PARENT": "<data_nvme>/.../checkpoint.pt",
                      "TRAINER_ARGS": ["--native-graphs", "...", "--held-native-graphs", "...", "--out",
                                       "/work/diloco-learner", "--continue-from", "<data_nvme>/.../checkpoint.pt"],
                      "COORDINATOR_BUNDLE": "...json", "LEARNER_BUNDLE": "...json", "RECIPE": "...", "STAGE": "..."},
     "overrides": {"diloco-learner.job.json": {"resources": {"gpu": 1, "memory": "80Gi"}},
                   "diloco-sft.run.json": {"settings": {"workers": 2}}}}

A string equal to "@@NAME@@" whose value is a list is spliced into the surrounding list (trainer arguments); other
occurrences are replaced textually. OUT is filled in automatically. Overrides merge into the filled template
(objects merge key by key, everything else replaces). Any placeholder left unfilled is an error, for the templates
listed in VALUES "files" (default: all of them).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import job as jobs  # noqa: E402

PLACEHOLDER = re.compile(r'@@([A-Z0-9_]+)@@')


def fill(value, values: dict):
    if isinstance(value, dict):
        return {key: fill(item, values) for key, item in value.items()}
    if isinstance(value, list):
        out = []
        for item in value:
            whole = PLACEHOLDER.fullmatch(item) if isinstance(item, str) else None
            if whole and isinstance(values.get(whole.group(1)), list):
                out.extend(values[whole.group(1)])
            else:
                out.append(fill(item, values))
        return out
    if isinstance(value, str):
        return PLACEHOLDER.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), value)
    return value


def merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        return {**base, **{key: merge(base.get(key), item) for key, item in override.items()}}
    return override


def render(values: dict, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    placeholders = {**values.get('placeholders', {}), 'OUT': str(out.resolve())}
    names = values.get('files') or sorted(p.name for p in HERE.glob('*.json'))
    written = []
    for name in names:
        filled = merge(fill(json.loads((HERE / name).read_text()), placeholders), values.get('overrides', {}).get(name, {}))
        left = sorted(set(PLACEHOLDER.findall(json.dumps(filled))))
        if left:
            raise SystemExit(f'{name}: unfilled placeholders {", ".join(left)}')
        if name.endswith('.job.json'):
            jobs.validate(filled)
        target = out / name
        target.write_text(json.dumps(filled, indent=1) + '\n')
        written.append(target)
    return written


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    for path in render(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])):
        print(path)
