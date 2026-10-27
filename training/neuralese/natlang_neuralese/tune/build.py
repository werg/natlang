"""`python -m natlang_neuralese.tune.build --plan plan.json --out DIR`: one resumable joint program fit.

Plan: ``{"manifest": <natlang.adapters/v1 object or path>, "engine": {"checkpoint", "device", "dtype"},
"rows": <JSONL path of {binding, graph, evidence_weight, occurrence}>, "recipe": {...Recipe}}``. Rows are
already split-admitted training rows; this entry never selects or reads development/held data.

Writes every referenced native tensor artifact (coefficients, bases, optimizer states) to ``artifacts/`` and
``state.json`` after every logical update; rerunning the same command restores them and resumes. ``plan.artifacts``
lists further directories to import first (published family versions). Finally ``result.json`` holds the release
fragment (`artifacts`, `memberships`) and the training receipt. ``plan.emitBoundaries`` appends one free-tune target
boundary per committed update to ``boundaries.jsonl`` for the learned optimizer (plans/neuralese/JOINT_LEARNED_OPTIMIZER.md).
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
    from .archive_links import read_archive, remember
    for path in sorted(Path(directory).glob('*.nnt')):
        body, reference = read_archive(path)
        artifact, ident = decode_native_tensor_artifact(body)
        if engine.store.put_native_tensor_artifact(artifact).id != ident or path.stem != ident:
            raise ValueError(f'artifact file {path} does not hold its named content')
        remember(engine, ident, reference)


def persist_artifacts(engine, ids, directory, states=None):
    """Write each artifact and its source closure once; content addressing makes rewrites unnecessary.

    Native states (`nzs1_`) go to ``states`` with their full derivation closure (parents, gradients, updates and
    optimizer histories), so the tuned soft states and their immutable lineage can be restored and verified."""
    from ..model.native_artifact import encode_native_tensor_artifact
    from .archive_links import link_verified, remember, signature
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
            if not path.exists() and not link_verified(engine, ident, path):
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(encode_native_state(record.block))
                os.replace(temporary, path)
                remember(engine, ident, (path.resolve(), signature(path)))
            pending.extend(derivation_dependencies(record.block.derivation))
            continue
        path = directory / f'{ident}.nnt'
        record = engine.store.get_native_tensor_artifact(ident)
        if record is None:
            raise ValueError(f'trainer state references unavailable artifact {ident}')
        if not path.exists() and not link_verified(engine, ident, path):
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(encode_native_tensor_artifact(record.artifact))
            os.replace(temporary, path)
            remember(engine, ident, (path.resolve(), signature(path)))
        if record.artifact.source_artifact_id:
            pending.append(record.artifact.source_artifact_id)


def import_states(engine, directory):
    """Restore complete archived histories with one full verification per terminal."""
    from ..serve.store import decode_native_state
    from .archive_links import read_archive, remember
    pending = {}
    references = {}
    for path in sorted(Path(directory).glob('*.nzs')):
        body, reference = read_archive(path)
        block, ident = decode_native_state(body)
        if path.stem != ident:
            raise ValueError(f'state file {path} does not hold its named content')
        pending[ident] = block
        references[ident] = reference
    records = engine.store.put_native_states(pending.values())
    if [record.id for record in records] != list(pending):
        raise ValueError('archived native states changed their content IDs')
    for ident, reference in references.items():
        remember(engine, ident, reference)


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
    recipe = Recipe.from_json(plan.get('recipe', {}))
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
    # Published family versions are part of the release: archived beside the fit's own residuals.
    family_artifacts = [item['artifact'] for item in manifest.get('families', [])]
    save = lambda: (persist_artifacts(engine, [*trainer.state['node_artifacts'].values(), *trainer.state['optimizer_states'].values(),
                                               *family_artifacts,
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
    rows = lower_rows(engine, rows, prompt_weight=recipe.prompt_weight,
                      soft={key: record['state'] for key, record in soft.items()}) if rows else []
    parent = json.loads((parent_dir / 'state.json').read_text()) if parent_dir is not None and state is None else None
    # Plan §8 incremental builds: `pinned` nodes stay frozen at earlier artifacts, `warm` nodes start from them (both
    # imported from the plan's `artifacts` directories, e.g. the active release's).
    trainer = JointTrainer(engine, manifest, rows + rollouts, recipe, state=state,
                           workdir=out / 'rl', parent=parent, pinned=plan.get('pinned'), warm=plan.get('warm'))
    save()
    while trainer.state['step'] < trainer.total_steps:
        before = json.loads(json.dumps(trainer.state)) if plan.get('emitBoundaries') else None
        record = trainer.step()
        save()
        if before is not None:
            # Free-tune target boundaries for the learned optimizer (tune/learned/targets.py), after the commit.
            from .learned.targets import tuner_boundary
            with (out / 'boundaries.jsonl').open('a') as stream:
                stream.write(json.dumps(tuner_boundary(trainer, before_state=before, label=plan.get('label', 'fit')),
                                        sort_keys=True) + '\n')
        print(json.dumps({'step': record['step'], 'objective': record['objective'], 'lr_factor': record['lr_factor'],
                          'mean_row_loss': sum(record['row_losses']) / max(len(record['row_losses']), 1)}), flush=True)
    fragment = trainer.manifest_fragment()
    families = [dict(item) for item in manifest.get('families', [])]
    if 'gguf' in plan.get('exports', []):
        export_nodes(trainer, engine, plan, out, fragment['artifacts'])
        export_families(trainer, engine, plan, out, families)
    soft_states = trainer.soft_fragment(soft) if soft else []
    if soft_states and 'gguf' in plan.get('exports', []):
        # The managed llama.cpp fork consumes soft states as exact native state spans (`--native-state`).
        from .soft_export import export_soft_states
        export_soft_states(engine, soft_states, out, base_gguf_sha256=plan['student'].get('baseGgufSha256') or '')
    _write(out / 'result.json', {'schema': 'natlang.tune.build-result/1', **fragment, 'families': families, 'softStates': soft_states,
                                 'softStateFiles': {item['state']: f"states/{item['state']}.nzs" for item in soft_states},
                                 'receipt': trainer.receipt()})
    return 0


def export_families(trainer, engine, plan, out, families):
    """One unscaled GGUF per published family version this program uses; the binding's gate is the request scale."""
    import dataclasses
    for family in families:
        membership = next((item for item in trainer.memberships
                           if (item['family'], item['version']) == (family['family'], family['version'])), None)
        if membership is None:
            continue  # a family without members in this program is not served by it
        members, bases = trainer._members([('family', membership)])
        _export_gguf(trainer.targets, engine, plan, out, family, [dataclasses.replace(member, scale=1.) for member in members], bases)


def export_nodes(trainer, engine, plan, out, artifacts):
    for artifact in artifacts:
        _export_gguf(trainer.targets, engine, plan, out, artifact, *trainer.node_members(artifact['node']))


def export_release_nodes(engine, plan, out, artifacts):
    """Per-node GGUF exports of a release fragment's stored adapters (e.g. a learned-optimizer release), through the
    same exact stack export as tuner nodes."""
    from .targets import module_map
    from .trainer import artifact_member
    for artifact in artifacts:
        bases = {}
        member = artifact_member(engine, artifact['artifact'], artifact['node'], 'lambda', 1., bases)
        targets, _ = module_map(sorted(member.deltas))
        _export_gguf(targets, engine, plan, out, artifact, [member], bases)


def _export_gguf(targets, engine, plan, out, artifact, members, bases):
    """One unscaled GGUF LoRA per scope node or family: llama.cpp sums the loaded residuals of a call's stack with its
    per-request scales, which equals the exported-stack sum exactly (§5.3). Each export binds the base GGUF."""
    from ..common.hashing import sha256_file_hex
    from ..model.native_adapters import target_parameters
    from .export import export_gguf, export_peft
    from .stack import stack_factors
    student = plan['student']
    if not student.get('hfConfigDir') or not student.get('baseGgufSha256'):
        raise ValueError('GGUF export needs student.hfConfigDir and student.baseGgufSha256')
    shapes = {name: tuple(value.shape) for name, value in target_parameters(engine.shared_model).items() if name in targets}
    file = f"exports/{artifact['artifact']}.gguf"
    target = out / file
    if not target.exists():
        peft = out / 'exports' / (artifact['artifact'] + '.peft')
        export_peft(stack_factors(members, bases, shapes), targets, peft, base_identity=dict(engine.native_identity),
                    base_model_name_or_path=student['hfConfigDir'], stack_identity=artifact['artifact'])
        export_gguf(peft, student['hfConfigDir'], target)
    artifact['exports'] = {'gguf': {'file': file, 'sha256': sha256_file_hex(target), 'baseGgufSha256': student['baseGgufSha256']}}


if __name__ == '__main__':
    raise SystemExit(main())
