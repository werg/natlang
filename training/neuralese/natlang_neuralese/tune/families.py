"""Cross-program family discovery and family fits (plans/HIERARCHICAL_LORA_TUNING.md §5.4, §5.5).

Discovery compares bindings of several programs by their **effective deltas** (each binding's whole lexical stack,
families excluded), with gauge-invariant Gram products (`similarity.delta_cosine`): cosine over raw factors is invalid
because ``BA = (BQ)(Q^-1 A)``. Programs must share the exact base. Proposals are crisp single-linkage clusters at a
threshold that span at least two programs, merged with author tags; choosing which proposals to fit is policy and
lives in the caller (`proposeFamilies`).

A family fit trains one shared residual on its members' training rows against the frozen base, with every private
program/folder/lambda residual absent, so the family learns behavior the private adapters already carried instead of
adding a second copy on top. It reuses the joint trainer: one tree node, the member bindings (namespaced by program)
under it, fixed membership weights and balanced per-program sampling weights. The trained family is then normalized to
its fixed amplitude ``rho_f = ||D_f||_q`` (the seed fit's own norm for a new family; the recorded amplitude for a new
version) and published as an immutable coefficient artifact with its `amplitudeHex`. A near-zero fit is absent.

`python -m natlang_neuralese.tune.families discover --plan P --out DIR` writes ``discovery.json``;
`python -m natlang_neuralese.tune.families fit --plan P --out DIR` writes ``result.json`` and ``artifacts/`` (plus a
GGUF export when the plan asks for one).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from . import TuneMathError
from .gates import family_norm
from .similarity import delta_cosine
from .stack import stack_factors
from .stack_export import load_artifacts, members_from_artifacts

FAMILY_NODE = 'tree:'


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(temporary, path)


# -- discovery ----------------------------------------------------------------------------------------------------
def binding_deltas(release):
    """``{binding_id: {target: (B, A)}}`` effective deltas of each tuned binding of one release directory, and the
    q weights of its targets. Families are excluded: discovery looks at what the private stack learned."""
    release = Path(release)
    manifest = json.loads((release / 'natlang.adapters.json').read_text())
    artifacts = load_artifacts(release / 'artifacts')
    by_node = {item['node']: item['artifact'] for item in manifest['artifacts']}
    scales = manifest.get('scales', {})
    deltas, q = {}, {}
    for binding in manifest['bindings']:
        entries = [{'id': by_node[node], 'scale': scales.get(node, 1), 'node': node}
                   for node in binding['ancestry'] if node in by_node and scales.get(node, 1) != 0]
        if not entries:
            continue
        members, bases, shapes = members_from_artifacts(artifacts, entries)
        factors = stack_factors(members, bases, shapes)
        effective = {name: (item.B.double(), item.A.double()) for name, item in factors.items() if item.B is not None}
        if effective:
            deltas[binding['id']] = effective
            q.update({name: 1. / float(torch.Size(shape).numel()) for name, shape in shapes.items()})
    return manifest, deltas, q


def discover(programs, *, threshold=.5, min_programs=2, tags=None):
    """Pairwise effective-delta cosines across programs and crisp threshold clusters, merged with author tags.

    ``programs``: ``[{"id", "release"}]``; ``tags``: ``{family: [[program, binding], ...]}``."""
    base, members, q = None, {}, {}
    for program in programs:
        manifest, deltas, weights = binding_deltas(program['release'])
        if base is None:
            base = manifest['base']
        elif manifest['base'] != base:
            raise TuneMathError(f'program {program["id"]} is tuned on another base; families never mix bases')
        q.update(weights)
        for binding, delta in deltas.items():
            members[(program['id'], binding)] = delta
    keys = sorted(members)
    similarities = []
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if left[0] == right[0]:
                continue  # within one program the shared ancestors already carry common behavior
            cosine = float(delta_cosine(members[left], members[right], q))
            similarities.append({'a': list(left), 'b': list(right), 'cosine': cosine})
    # Single-linkage clusters over pairs at or above the threshold.
    parent = {key: key for key in keys}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for item in similarities:
        if item['cosine'] >= threshold:
            parent[find(tuple(item['a']))] = find(tuple(item['b']))
    groups = {}
    for key in keys:
        groups.setdefault(find(key), []).append(list(key))
    clusters = [sorted(group) for group in groups.values() if len({program for program, _ in group}) >= min_programs]
    tagged = {name: sorted([list(member) for member in group]) for name, group in sorted((tags or {}).items())}
    for name, group in tagged.items():
        unknown = [member for member in group if tuple(member) not in members]
        if unknown:
            raise TuneMathError(f'author family {name} names bindings without a tuned release: {unknown}')
    return {'schema': 'natlang.tune-family-discovery/v1', 'base': base, 'threshold': threshold,
            'min_programs': min_programs, 'members': [list(key) for key in keys], 'similarities': similarities,
            'clusters': sorted(clusters), 'tags': tagged}


# -- family fit ---------------------------------------------------------------------------------------------------
def family_manifest(base, members):
    """A one-node scaffold: every member binding (``program/binding``) has only the shared family node."""
    from ..common.hashing import canonical_json_sha256_hex
    bindings = [{'id': member, 'component': member, 'definitionId': None, 'contextId': None, 'origin': 'named',
                 'ancestry': [FAMILY_NODE], 'semanticHash': canonical_json_sha256_hex({'family-member': member})}
                for member in sorted(members)]
    return {'schema': 'natlang.adapters/v1', 'program': {'id': 'natlang.tune-family', 'buildHash': ''}, 'base': base,
            'tree': [{'id': FAMILY_NODE, 'level': 'tree', 'path': '', 'parent': None}], 'bindings': bindings,
            'artifacts': [], 'families': [], 'memberships': [], 'softStates': [], 'scales': {}}


def balance_rows(rows):
    """Equal total weight per program: a program with many rows or lambdas does not own the family. The trainer's
    batch objective already averages over bindings; this fixes the program share."""
    per_program = {}
    for row in rows:
        program = row['binding'].split('/', 1)[0]
        per_program[program] = per_program.get(program, 0.) + row['evidence_weight']
    if not per_program or any(total <= 0 for total in per_program.values()):
        raise TuneMathError('every family program needs positively weighted training rows')
    mean = sum(per_program.values()) / len(per_program)
    return [{**row, 'evidence_weight': row['evidence_weight'] * mean / per_program[row['binding'].split('/', 1)[0]]}
            for row in rows]


def normalized_family(trainer, *, rho=None, absent_below=1e-12):
    """The trained family node normalized to amplitude ``rho`` (its own norm when None), as a new immutable
    coefficient artifact. Returns ``(artifact_id, rho)`` or ``(None, norm)`` for an absent family."""
    from ..model.native_artifact import NativeTensorArtifact
    engine = trainer.engine
    artifact_id = trainer.state['node_artifacts'][FAMILY_NODE]
    record = engine.store.get_native_tensor_artifact(artifact_id).artifact
    factors = trainer._node_factors(artifact_id, record.tensors)
    norm = float(family_norm(factors, trainer.q))
    if norm < absent_below:
        return None, norm
    rho = norm if rho is None else float(rho)
    if not rho > 0:
        raise TuneMathError('a family amplitude must be positive')
    scale = rho / norm
    spec = record.provenance['adapter_spec']
    # The delta is linear in the core (native subspace) and in B (free LoRA): scaling them scales D_f exactly.
    tensors = {key: value * scale if spec['kind'] != 'free-lora' or key.startswith('lora_B.') else value
               for key, value in record.tensors.items()}
    published = engine.store.put_native_tensor_artifact(NativeTensorArtifact('adapter-coefficients', engine.native_identity,
        tensors, record.source_artifact_id, {**record.provenance, 'family_amplitude_hex': rho.hex(),
                                             'family_trained_artifact': artifact_id}), trainer.owner)
    return published.id, rho


def fit(plan, out):
    """One family fit from a plan: ``{family, members, rows, student|engine, recipe, rho?, exports?}``."""
    from ..serve import load_native_engine
    from .build import persist_artifacts, export_nodes
    from .rows import lower_rows
    from .trainer import JointTrainer, Recipe
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = plan.get('engine') or {'checkpoint': plan['student']['checkpoint'], **plan['student'].get('engine', {})}
    engine = load_native_engine(config['checkpoint'], device=config.get('device', 'cpu'),
                                dtype=getattr(torch, config.get('dtype', 'float32')))
    members = sorted(set(plan['members']))
    turns = [json.loads(line) for line in Path(plan['rows']).read_text().splitlines() if line.strip()]
    if {turn['binding'] for turn in turns} - set(members):
        raise TuneMathError('family rows name bindings outside the family')
    rows = balance_rows(lower_rows(engine, turns, prompt_weight=plan.get('recipe', {}).get('prompt_weight', .1)))
    recipe = {key: value for key, value in plan.get('recipe', {}).items() if key != 'prompt_weight'}
    rank = recipe.pop('rank', 8)
    recipe = Recipe.from_json({**recipe, 'ranks': {'tree': rank, 'folder': 0, 'lambda': 0}})
    manifest = family_manifest(dict(engine.native_identity), members)
    state_path = out / 'state.json'
    from .build import import_archive
    import_archive(engine, [out / 'artifacts'], [out / 'states'])
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    trainer = JointTrainer(engine, manifest, rows, recipe, state=state)
    save = lambda: (persist_artifacts(engine, [*trainer.state['node_artifacts'].values(),
                                               *trainer.state['optimizer_states'].values()], out / 'artifacts'),
                    _write(state_path, trainer.state))
    save()
    while trainer.state['step'] < trainer.total_steps:
        record = trainer.step()
        save()
        print(json.dumps({'step': record['step'], 'mean_row_loss': sum(record['row_losses']) / max(len(record['row_losses']), 1)}),
              flush=True)
    rho = plan.get('rho')
    artifact, rho = normalized_family(trainer, rho=float.fromhex(rho) if isinstance(rho, str) else rho)
    receipt = trainer.receipt()
    if artifact is None:
        _write(out / 'result.json', {'schema': 'natlang.tune-family-fit/v1', 'family': plan['family'], 'absent': True,
                                     'norm': rho, 'receipt': receipt})
        return None
    persist_artifacts(engine, [artifact], out / 'artifacts')
    from ..common.hashing import canonical_json_sha256_hex
    version = 'v' + canonical_json_sha256_hex({'artifact': artifact, 'amplitude': rho.hex()})[:12]
    result = {'schema': 'natlang.tune-family-fit/v1', 'family': plan['family'], 'version': version, 'artifact': artifact,
              'amplitudeHex': rho.hex(), 'members': members, 'rank': rank, 'receipt': receipt}
    if 'gguf' in plan.get('exports', []):
        # The published family is served like any scope residual: one unscaled GGUF, its gate is the request scale.
        exported = [{'node': FAMILY_NODE, 'artifact': artifact}]
        trainer.state['node_artifacts'][FAMILY_NODE] = artifact
        export_nodes(trainer, engine, plan, out, exported)
        result['exports'] = exported[0]['exports']
    _write(out / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('discover', 'fit'))
    parser.add_argument('--plan', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == 'discover':
        _write(out / 'discovery.json', discover(plan['programs'], threshold=plan.get('threshold', .5),
                                                min_programs=plan.get('minPrograms', 2), tags=plan.get('tags')))
    else:
        fit(plan, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
