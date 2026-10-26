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


def persist_artifacts(engine, ids, directory, states=None):
    """Write each artifact and its source closure once; content addressing makes rewrites unnecessary.

    Native states (`nzs1_`) go to ``states`` with their full derivation closure (parents, gradients, updates and
    optimizer histories), so the tuned soft states and their immutable lineage can be restored and verified."""
    from ..model.native_artifact import encode_native_tensor_artifact
    directory.mkdir(parents=True, exist_ok=True)
    pending = [ident for ident in ids if ident]
    seen = set()
    while pending:
        ident = pending.pop()
        if ident in seen:
            continue
        seen.add(ident)
        if ident.startswith('nzs1_'):
            if states is None:
                continue  # a state-gradient's parent; persisted by the caller that owns states
            from ..model.native_derivative_contract import derivation_dependencies
            from ..serve.store import encode_native_state
            record = engine.store.get_native_state(ident)
            if record is None:
                raise ValueError(f'trainer state references unavailable native state {ident}')
            states.mkdir(parents=True, exist_ok=True)
            path = states / f'{ident}.nzs'
            if not path.exists():
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(encode_native_state(record.block))
                os.replace(temporary, path)
            pending.extend(derivation_dependencies(record.block.derivation))
            continue
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


def import_states(engine, directory):
    """Restore archived states parents first (each one's derivation names its parent)."""
    from ..serve.store import decode_native_state
    pending = {}
    for path in sorted(Path(directory).glob('*.nzs')):
        block, ident = decode_native_state(path.read_bytes())
        if path.stem != ident:
            raise ValueError(f'state file {path} does not hold its named content')
        pending[ident] = block
    while pending:
        ready = [ident for ident, block in pending.items()
                 if block.derivation is None or block.derivation.get('parent_state_id') not in pending]
        if not ready:
            raise ValueError('archived native states have a cyclic derivation')
        for ident in sorted(ready):
            if engine.store.put_native_state(pending.pop(ident)).id != ident:
                raise ValueError(f'archived state {ident} changed its content ID')


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    manifest = plan['manifest']
    if isinstance(manifest, str):
        manifest = json.loads((Path(path).parent / manifest).read_text())
    read = lambda name: [json.loads(line) for line in (Path(path).parent / name).read_text().splitlines() if line.strip()]
    rows = read(plan['rows']) if plan.get('rows') else []
    # §5.1a: scored student rollouts (natlang.tune-rollout/v1) are the RL rows of the same fit.
    return plan, manifest, rows, read(plan['rollouts']) if plan.get('rollouts') else []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--plan', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    plan, manifest, rows, rollouts = load_plan(args.plan)
    # A continued round names the previous round's build directory: its state, versions and soft initialization.
    parent_dir = Path(args.plan).parent / plan['parent'] if plan.get('parent') else None
    from ..serve import load_native_engine
    config = plan.get('engine') or {'checkpoint': plan['student']['checkpoint'], **plan['student'].get('engine', {})}
    engine = load_native_engine(config['checkpoint'], device=config.get('device', 'cpu'),
                                dtype=getattr(torch, config.get('dtype', 'float32')))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for directory in [*(Path(args.plan).parent / item for item in plan.get('artifacts', [])),
                      *([parent_dir / 'artifacts'] if parent_dir else []), out / 'artifacts']:
        if directory.is_dir():
            import_artifacts(engine, directory)
    for directory in [*([parent_dir / 'states'] if parent_dir else []), out / 'states']:
        if directory.is_dir():
            import_states(engine, directory)
    state_path = out / 'state.json'
    save = lambda: (persist_artifacts(engine, [*trainer.state['node_artifacts'].values(), *trainer.state['optimizer_states'].values(),
                                               *trainer.state.get('soft_states', {}).values(),
                                               *trainer.state.get('soft_optimizer_states', {}).values()],
                                      out / 'artifacts', out / 'states'), _write(state_path, trainer.state))
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    soft = {}
    if plan.get('softStates'):
        # Plan §3.3: initialize once (in context, under the base = the zero-residual initial stack); a resumed run
        # reuses the recorded initial states rather than writing them again.
        from .soft import initialize_soft_states
        init_path = out / 'soft-init.json'
        if not init_path.exists() and parent_dir is not None and (parent_dir / 'soft-init.json').exists():
            _write(init_path, json.loads((parent_dir / 'soft-init.json').read_text()))
        soft = json.loads(init_path.read_text()) if init_path.exists() else initialize_soft_states(engine, rows)
        if not init_path.exists():
            persist_artifacts(engine, [record['state'] for record in soft.values()], out / 'artifacts', out / 'states')
            _write(init_path, soft)
    from .rows import lower_rows
    rows = lower_rows(engine, rows, prompt_weight=plan.get('recipe', {}).get('prompt_weight', .1),
                      soft={key: record['state'] for key, record in soft.items()}) if rows else []
    parent = json.loads((parent_dir / 'state.json').read_text()) if parent_dir is not None and state is None else None
    trainer = JointTrainer(engine, manifest, rows + rollouts, Recipe.from_json(plan.get('recipe', {})), state=state,
                           workdir=out / 'rl', parent=parent)
    save()
    while trainer.state['step'] < trainer.total_steps:
        record = trainer.step()
        save()
        print(json.dumps({'step': record['step'], 'objective': record['objective'], 'lr_factor': record['lr_factor'],
                          'mean_row_loss': sum(record['row_losses']) / max(len(record['row_losses']), 1)}), flush=True)
    fragment = trainer.manifest_fragment()
    if 'gguf' in plan.get('exports', []):
        export_nodes(trainer, engine, plan, out, fragment['artifacts'])
    soft_states = trainer.soft_fragment(soft) if soft else []
    _write(out / 'result.json', {'schema': 'natlang.tune.build-result/1', **fragment, 'families': [], 'softStates': soft_states,
                                 'softStateFiles': {item['state']: f"states/{item['state']}.nzs" for item in soft_states},
                                 'receipt': trainer.receipt()})
    return 0


def export_nodes(trainer, engine, plan, out, artifacts):
    """One GGUF LoRA per scope node, unscaled: llama.cpp sums the loaded residuals of a call's stack with its
    per-request scales, which equals the exported-stack sum exactly (§5.3). Each export binds the base GGUF."""
    from ..common.hashing import sha256_file_hex
    from ..model.native_adapters import target_parameters
    from .export import export_gguf, export_peft
    from .stack import stack_factors
    student = plan['student']
    if not student.get('hfConfigDir') or not student.get('baseGgufSha256'):
        raise ValueError('GGUF export needs student.hfConfigDir and student.baseGgufSha256')
    shapes = {name: tuple(value.shape) for name, value in target_parameters(engine.shared_model).items() if name in trainer.targets}
    for artifact in artifacts:
        file = f"exports/{artifact['artifact']}.gguf"
        target = out / file
        if not target.exists():
            members, bases = trainer.node_members(artifact['node'])
            peft = out / 'exports' / (artifact['artifact'] + '.peft')
            export_peft(stack_factors(members, bases, shapes), trainer.targets, peft, base_identity=dict(engine.native_identity),
                        base_model_name_or_path=student['hfConfigDir'], stack_identity=artifact['artifact'])
            export_gguf(peft, student['hfConfigDir'], target)
        artifact['exports'] = {'gguf': {'file': file, 'sha256': sha256_file_hex(target), 'baseGgufSha256': student['baseGgufSha256']}}


if __name__ == '__main__':
    raise SystemExit(main())
