"""`python -m natlang_neuralese.tune.build --plan plan.json --out DIR`: one resumable joint program fit.

Plan: ``{"manifest": <natlang.adapters/v1 object or path>, "engine": {"checkpoint", "device", "dtype"},
"rows": <JSONL path of {binding, graph, evidence_weight, occurrence}>, "recipe": {...Recipe}}``. Rows are
already split-admitted training rows; this entry never selects or reads development/held data.

Writes every referenced native tensor artifact (coefficients, bases, optimizer states) to ``artifacts/`` and
``state.json`` after every logical update; rerunning the same command restores them and resumes. ``plan.artifacts``
lists further directories to import first (published family versions). Finally ``result.json`` holds the release
fragment (`artifacts`, `memberships`) and the training receipt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .trainer import JointTrainer, Recipe


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(temporary, path)


def import_artifacts(engine, directory):
    from ..model.native_artifact import decode_native_tensor_artifact
    for path in sorted(Path(directory).glob('*.nnt')):
        artifact, ident = decode_native_tensor_artifact(path.read_bytes())
        if engine.store.put_native_tensor_artifact(artifact).id != ident or path.stem != ident:
            raise ValueError(f'artifact file {path} does not hold its named content')


def persist_artifacts(engine, ids, directory):
    """Write each artifact and its source closure once; content addressing makes rewrites unnecessary."""
    from ..model.native_artifact import encode_native_tensor_artifact
    directory.mkdir(parents=True, exist_ok=True)
    pending = [ident for ident in ids if ident]
    while pending:
        ident = pending.pop()
        path = directory / f'{ident}.nnt'
        record = engine.store.get_native_tensor_artifact(ident)
        if record is None:
            raise ValueError(f'trainer state references unavailable artifact {ident}')
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(encode_native_tensor_artifact(record.artifact))
            os.replace(temporary, path)
        if record.artifact.source_artifact_id:
            pending.append(record.artifact.source_artifact_id)


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    manifest = plan['manifest']
    if isinstance(manifest, str):
        manifest = json.loads((Path(path).parent / manifest).read_text())
    rows_path = Path(path).parent / plan['rows']
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    return plan, manifest, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    plan, manifest, rows = load_plan(args.plan)
    from ..serve import load_native_engine
    config = plan['engine']
    engine = load_native_engine(config['checkpoint'], device=config.get('device', 'cpu'),
                                dtype=getattr(torch, config.get('dtype', 'float32')))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for directory in [*(Path(args.plan).parent / item for item in plan.get('artifacts', [])), out / 'artifacts']:
        if directory.is_dir():
            import_artifacts(engine, directory)
    state_path = out / 'state.json'
    save = lambda: (persist_artifacts(engine, [*trainer.state['node_artifacts'].values(), *trainer.state['optimizer_states'].values()],
                                      out / 'artifacts'), _write(state_path, trainer.state))
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    trainer = JointTrainer(engine, manifest, rows, Recipe.from_json(plan.get('recipe', {})), state=state)
    save()
    while trainer.state['step'] < trainer.recipe.steps:
        record = trainer.step()
        save()
        print(json.dumps({'step': record['step'], 'mean_row_loss': sum(record['row_losses']) / len(record['row_losses'])}), flush=True)
    _write(out / 'result.json', {'schema': 'natlang.tune.build-result/1', 'receipt': trainer.receipt()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
