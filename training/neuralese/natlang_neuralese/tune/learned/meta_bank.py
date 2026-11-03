"""Continuous meta-learning of the shared LoRA bank (plans/neuralese/CONTINUOUS_BANK_METALEARNING_20261103.md).

Fast weights are one program's component-token mixture over the bank (`mixture`: free logits and group gains, the
law the learned optimizer and deployment execute); slow weights are the bank's component factors and shared basis
(`bank.ComponentBank`). One outer step:

1. sample ``meta_batch`` meta-training families; per family disjoint support and query worlds from its train worlds;
2. inner step: `fit.fit_in_bank` on the support rows with the live bank held fixed (coefficients only);
3. outer step: the query rows' teacher-forced loss at the fitted coefficients (held constant: first order), backward
   into the live component factors and basis; Adam; then the publication invariants are restored exactly (basis
   re-orthonormalized with the factors re-expressed, every component back at unit relative energy).

Evaluation (`evaluate`) fits each held family on its train worlds and scores its development (or confirmation)
worlds under four arms at matched adaptation compute (the same number of support forward/backward passes): the
unadapted parent, the meta-trained bank, the identical bank before meta-training, and a plain per-program free LoRA
(the tuner's ``free-lora`` parameterization). Held families never enter meta-training.

Supervised episodes need verified trajectories: `graph_rows` reads native graph sources (verified teacher programs,
one world per invocation) and `rollout_rows` lowers oracle-checked successful RL rollouts of the parent (one world per
task). Both pass the protected-worlds union before they enter a pool.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError

POOL_SCHEMA = 'natlang.bank-meta-pool/1'
STATE_SCHEMA = 'natlang.bank-meta-state/1'
EVAL_SCHEMA = 'natlang.bank-meta-evaluation/1'
SCOPE = 'program'
MANIFEST = {'bindings': [{'id': SCOPE, 'ancestry': [SCOPE]}], 'scales': {}}
ARMS = ('parent', 'meta', 'initial', 'lora')


# -- episode supply -----------------------------------------------------------------------------------------------
def _lines(path):
    path = Path(path)
    with path.open('rb') as handle:
        if path.suffix == '.zst':
            import zstandard
            yield from io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(handle), encoding='utf-8')
        else:
            yield from io.TextIOWrapper(handle, encoding='utf-8')


def _graph_tokens(graph):
    return sum(len(node.get('token_ids') or ()) for node in graph['nodes'])


def _targets(graph):
    return sum(1 for node in graph['nodes'] for weight in node.get('target_weights') or () if weight > 0)


def graph_rows(paths, pattern, *, protected, max_tokens, prefix=None, families=None, index_only=False):
    """Rows of verified native graph sources whose id matches ``pattern`` (named groups ``family`` and ``world``).
    ``prefix``: cheap substring every candidate line contains; ``families``: keep only these; ``index_only``: rows
    carry the graph id instead of the graph (a cheap first pass over large sources). Protected-related, non-train,
    oversize and target-free graphs are skipped and counted."""
    regex = re.compile(pattern)
    counts = {'read': 0, 'matched': 0, 'protected': 0, 'oversize': 0, 'no_target': 0, 'split': 0}
    rows = []
    for path in paths:
        for line in _lines(path):
            counts['read'] += 1
            if prefix and prefix not in line[:600]:
                continue
            graph = json.loads(line)
            match = regex.match(graph.get('id', ''))
            if match is None:
                continue
            if families is not None and match.group('family') not in families:
                continue
            counts['matched'] += 1
            if graph.get('split') != 'train':
                counts['split'] += 1
                continue
            if protected is not None and protected.related({'id': graph['id'], 'source_groups': graph.get('source_groups') or []}):
                counts['protected'] += 1
                continue
            if _graph_tokens(graph) > max_tokens:
                counts['oversize'] += 1
                continue
            if not _targets(graph):
                counts['no_target'] += 1
                continue
            rows.append({'family': match.group('family'), 'world': match.group('world'),
                         'graph': {'id': graph['id']} if index_only else graph,
                         'actor': {'kind': 'verified-source', 'source': str(path)}})
    return rows, counts


def rollout_rows(engine, paths, *, protected, max_tokens, policy=None):
    """Successful oracle-checked rollouts as rows (family: the task family, world: the task). Every model turn of the
    episode is one node with its sampled reply as the only target; episodes that exchanged Neuralese state are left to
    the whole-graph path (counted, not lowered). ``policy``: the required actor checkpoint SHA-256, or None."""
    from ...data.native_graph_schema import SCHEMA as GRAPH_SCHEMA
    from ...serve.targets import target_items
    counts = {'read': 0, 'unsuccessful': 0, 'other_policy': 0, 'protected': 0, 'neuralese': 0, 'oversize': 0}
    rows = []
    for path in paths:
        for line in _lines(path):
            rollout = json.loads(line)
            counts['read'] += 1
            reward = rollout.get('reward') or {}
            if rollout.get('outcome') != 'task_success' or reward.get('quality') != 1.0 or rollout.get('split') != 'train':
                counts['unsuccessful'] += 1
                continue
            actor = (rollout.get('policy') or {}).get('checkpoint_sha256')
            if policy is not None and actor != policy:
                counts['other_policy'] += 1
                continue
            case = {'id': rollout['task_id'], 'source_groups': rollout.get('source_groups') or [],
                    'source_ids': rollout.get('source_ids') or []}
            if protected is not None and protected.related(case):
                counts['protected'] += 1
                continue
            nodes = []
            for index, turn in enumerate(rollout['turns']):
                prompt, target = target_items(engine, turn['messages'], turn.get('tools'), turn['target'])
                if any(item[0] != 'tok' for item in [*prompt, *target]):
                    nodes = None
                    break
                ids = [int(item[1]) for item in prompt] + [int(item[1]) for item in target]
                nodes.append({'name': f'turn:{index}', 'token_ids': ids,
                              'target_weights': [0.] * len(prompt) + [1.] * len(target)})
            if nodes is None:
                counts['neuralese'] += 1
                continue
            graph = {'schema': GRAPH_SCHEMA, 'id': f"rollout:{rollout['task_id']}:{rollout.get('group')}:{rollout.get('sample')}",
                     'split': 'train', 'source_groups': list(case['source_groups']),
                     'tokenizer_fingerprint': engine.model_config['tokenizer_fingerprint'], 'nodes': nodes}
            if _graph_tokens(graph) > max_tokens:
                counts['oversize'] += 1
                continue
            rows.append({'family': rollout['family'], 'world': rollout['task_id'], 'graph': graph,
                         'actor': {'kind': 'parent-rollout', 'checkpoint_sha256': actor, 'source': str(path)}})
    return rows, counts


def _order(value, salt):
    return hashlib.sha256(f'{salt}:{value}'.encode()).hexdigest()


def build_pool(rows, *, held_fraction=.25, support_worlds=3, query_worlds=2, dev_worlds=2, confirm_worlds=1,
               max_families=None, rows_per_world=1, seed=0):
    """Assign families to meta-training or held and worlds to splits.

    A held family's worlds split into train (support for its evaluation fits), dev (scored) and confirmation (scored
    once); a meta-training family contributes only train worlds, sampled into disjoint support/query sets per
    episode. Families are the held-out unit; world order within a family is its natural (numeric) order so later
    invocations score."""
    families = {}
    for row in rows:
        families.setdefault(row['family'], {}).setdefault(row['world'], []).append(row)
    def natural(world):
        return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', world)]
    pool = {'schema': POOL_SCHEMA, 'seed': seed, 'held_fraction': held_fraction,
            'worlds': {'support': support_worlds, 'query': query_worlds, 'dev': dev_worlds, 'confirmation': confirm_worlds},
            'families': {}}
    names = sorted(families, key=lambda name: _order(name, seed))
    for name in names:
        worlds = sorted(families[name], key=natural)
        held = int(_order(name, f'held-{seed}')[:8], 16) / 16 ** 8 < held_fraction
        need = support_worlds + dev_worlds + confirm_worlds if held else support_worlds + query_worlds
        if len(worlds) < need:
            continue
        if held:
            splits = (['train'] * support_worlds + ['dev'] * dev_worlds
                      + ['confirmation'] * (len(worlds) - support_worlds - dev_worlds))
        else:
            splits = ['train'] * len(worlds)
        pool['families'][name] = {'role': 'held' if held else 'meta-train', 'worlds': [
            {'world': world, 'split': split, 'rows': sorted(families[name][world], key=lambda r: r['graph']['id'])[:rows_per_world]}
            for world, split in zip(worlds, splits)]}
    if max_families is not None:
        for role in ('meta-train', 'held'):
            keep = [name for name in pool['families'] if pool['families'][name]['role'] == role]
            limit = max_families[role] if isinstance(max_families, dict) else max_families
            for name in keep[limit:]:
                del pool['families'][name]
    return pool


def write_pool(pool, directory):
    """``rows.jsonl`` (one graph per line) and ``pool.json`` naming each world's rows by graph id; returns the pool
    receipt with both hashes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lines, index = [], {'schema': POOL_SCHEMA, **{key: value for key, value in pool.items() if key not in ('schema', 'families')}, 'families': {}}
    for name, family in pool['families'].items():
        index['families'][name] = {'role': family['role'], 'worlds': []}
        for world in family['worlds']:
            ids = []
            for row in world['rows']:
                lines.append(json.dumps({'family': name, 'world': world['world'], 'actor': row['actor'], 'graph': row['graph']},
                                        sort_keys=True))
                ids.append(row['graph']['id'])
            index['families'][name]['worlds'].append({'world': world['world'], 'split': world['split'], 'graphs': ids})
    (directory / 'rows.jsonl').write_text(''.join(line + '\n' for line in lines))
    index['rows_sha256'] = hashlib.sha256((directory / 'rows.jsonl').read_bytes()).hexdigest()
    counts = {role: sum(1 for f in index['families'].values() if f['role'] == role) for role in ('meta-train', 'held')}
    index['counts'] = {**counts, 'rows': len(lines)}
    (directory / 'pool.json').write_text(json.dumps(index, indent=1, sort_keys=True))
    return {'pool': str(directory / 'pool.json'), 'pool_sha256': hashlib.sha256((directory / 'pool.json').read_bytes()).hexdigest(),
            **index['counts']}


def read_pool(directory):
    directory = Path(directory)
    index = json.loads((directory / 'pool.json').read_text())
    if index.get('schema') != POOL_SCHEMA:
        raise LearnedOptimizerError('not a bank meta-learning pool')
    if hashlib.sha256((directory / 'rows.jsonl').read_bytes()).hexdigest() != index['rows_sha256']:
        raise LearnedOptimizerError('pool rows changed after the pool was written')
    graphs = {}
    for line in (directory / 'rows.jsonl').read_text().splitlines():
        row = json.loads(line)
        graphs[row['graph']['id']] = row
    pool = {**index, 'families': {}}
    for name, family in index['families'].items():
        pool['families'][name] = {'role': family['role'], 'worlds': [
            {'world': world['world'], 'split': world['split'], 'rows': [graphs[g] for g in world['graphs']]}
            for world in family['worlds']]}
    return pool


# -- recipe and state ---------------------------------------------------------------------------------------------
@dataclass
class MetaRecipe:
    rank: int = 8                 # shared basis rank R per target matrix
    components: int = 16          # bank size K
    component_rank: int = 4       # rank of each component core inside the basis
    inner_steps: int = 6          # coefficient Adam steps on the free law
    refine_steps: int = 2         # then on the fixed top-p law (`fit.fit_in_bank`)
    inner_lr: float = .1
    outer_lr_bank: float = 1e-2
    outer_lr_basis: float = 1e-4
    meta_batch: int = 2
    support_worlds: int = 3
    query_worlds: int = 2
    train_bank: bool = True       # AGENTS.md "Learned LoRA mixtures and bases": explicit
    train_basis: bool = True
    lora_rank: int = 8            # the plain per-program LoRA baseline
    lora_lr: float = 1e-3
    target_pattern: str | None = None
    seed: int = 0

    def __post_init__(self):
        for name in ('rank', 'components', 'component_rank', 'meta_batch', 'support_worlds', 'query_worlds', 'lora_rank'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise LearnedOptimizerError(f'{name} must be a positive integer')
        if self.inner_steps + self.refine_steps < 1 or self.component_rank > self.rank:
            raise LearnedOptimizerError('the inner fit needs steps and components fit inside the basis')
        if self.train_bank is not True or self.train_basis is not True:
            raise LearnedOptimizerError('bank meta-learning trains the bank and basis explicitly')

    @property
    def fit_passes(self):
        return self.inner_steps + self.refine_steps

    def as_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, value):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in value.items() if k in known})


def random_components(bank, count, rank, seed):
    """``count`` seeded Gaussian components at unit relative energy: the un-meta-trained starting bank."""
    from .bank import Component, component_id, normalize
    generator = torch.Generator().manual_seed(seed)
    out = []
    for index in range(count):
        factors = {name: (torch.randn(*bank.basis.left[name].shape[:-2], bank.basis.rank, rank, generator=generator),
                          torch.randn(*bank.basis.left[name].shape[:-2], bank.basis.rank, rank, generator=generator))
                   for name in bank.basis.names}
        source = {'kind': 'random', 'seed': seed, 'index': index}
        component, _ = normalize(Component(component_id(source, factors), factors, None, f'random {index}', source,
                                           ('tree', 'folder', 'lambda')), bank.q)
        out.append(component)
    return out


def restore_invariants(bank, leaves, *, basis=True):
    """Exact publication invariants on live leaves: orthonormal basis columns with the factors re-expressed so every
    composed update is unchanged (``U P Q^T V^T = Q_u (R_u P)(R_v Q)^T Q_v^T``), then each component rescaled to unit
    relative energy (`bank.normalize`). Returns the per-component scales applied."""
    from ..gates import factor_sq_norm
    from .bank import orthonormalize
    with torch.no_grad():
        if basis:
            for name in bank.basis.names:
                left, right = leaves['basis']['left.' + name], leaves['basis']['right.' + name]
                q_left, r_left = orthonormalize(left)
                q_right, r_right = orthonormalize(right)
                left.copy_(q_left)
                right.copy_(q_right)
                for factors in leaves['factors_by_component'].values():
                    if name in factors:
                        P, Q = factors[name]
                        P.copy_(r_left @ P)
                        Q.copy_(r_right @ Q)
        scales = []
        for factors in leaves['factors_by_component'].values():
            total = sum(float(bank.q[name] * factor_sq_norm(P, Q.transpose(-2, -1))) for name, (P, Q) in factors.items())
            scale = 1. / math.sqrt(max(total, 1e-24))
            root = math.sqrt(scale)
            for P, Q in factors.values():
                P.mul_(root)
                Q.mul_(root)
            scales.append(scale)
    return scales


def _detached(leaves):
    return ({k: {name: (P.detach(), Q.detach()) for name, (P, Q) in v.items()} for k, v in leaves['factors_by_component'].items()},
            {key: value.detach() for key, value in leaves['basis'].items()})


def _snapshot_leaves(bank):
    from .trainer import _bank_leaves
    return _bank_leaves(bank, _TrainAll)


class _TrainAll:
    train_bank = True
    train_basis = True


class BankMetaLearner:
    """One operation's bank, live leaves, outer optimizer and history (``out/state.json`` + ``out/state.torch``)."""

    def __init__(self, engine, pool, recipe, out, *, parent_sha256):
        from .bank import new_bank, weight_svd_basis
        from .fit import placeholder_adapters
        from .mixture import MixtureFormat
        from ...model.native_adapters import create_adapter, target_parameters
        self.engine, self.pool, self.recipe, self.out = engine, pool, recipe, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        names = sorted(target_parameters(engine.shared_model))
        if recipe.target_pattern:
            names = [name for name in names if re.search(recipe.target_pattern, name)]
        if not names:
            raise LearnedOptimizerError('no target matrices selected')
        self.names = names
        torch.manual_seed(recipe.seed)
        bank = new_bank(engine, weight_svd_basis(engine, names, recipe.rank), MixtureFormat())
        self.bank0 = bank.with_components(random_components(bank, recipe.components, recipe.component_rank, recipe.seed),
                                          reason='random initialization')
        self.placeholder = placeholder_adapters(engine, self.bank0, [SCOPE])[SCOPE]
        self.lora = create_adapter(engine, {'kind': 'free-lora', 'rank': recipe.lora_rank, 'alpha': float(recipe.lora_rank),
                                            'seed': recipe.seed, 'parameter_names': names}).id
        self.leaves = _snapshot_leaves(self.bank0)
        self.optimizer = torch.optim.Adam([
            {'params': [t for f in self.leaves['factors_by_component'].values() for pair in f.values() for t in pair],
             'lr': recipe.outer_lr_bank},
            {'params': list(self.leaves['basis'].values()), 'lr': recipe.outer_lr_basis}])
        self.state = {'schema': STATE_SCHEMA, 'recipe': recipe.as_json(), 'parent_sha256': parent_sha256,
                      'parents': [{'sha256': parent_sha256, 'from_step': 0}],
                      'initial_bank': self.bank0.snapshot, 'targets': len(names), 'step': 0, 'history': [],
                      'evaluations': [], 'rng': recipe.seed}
        self.rng = random.Random(recipe.seed)
        self._restore()

    # -- persistence --------------------------------------------------------------------------------------------
    def _restore(self):
        path = self.out / 'state.json'
        if not path.exists():
            return
        state = json.loads(path.read_text())
        if state.get('schema') != STATE_SCHEMA or state['recipe'] != self.recipe.as_json():
            raise LearnedOptimizerError('the saved meta-learning state has another recipe')
        same_parent = state['parent_sha256'] == self.state['parent_sha256']
        if same_parent and state['initial_bank'] != self.bank0.snapshot:
            raise LearnedOptimizerError('the saved state started from another initial bank')
        tensors = torch.load(self.out / 'state.torch', map_location='cpu', weights_only=True)
        if tensors.get('step') != state['step']:
            raise LearnedOptimizerError('state tensors and record are from different steps')
        with torch.no_grad():
            for k, factors in self.leaves['factors_by_component'].items():
                for name, (P, Q) in factors.items():
                    P.copy_(tensors[f'component.{k}.P.{name}'])
                    Q.copy_(tensors[f'component.{k}.Q.{name}'])
            for key, value in self.leaves['basis'].items():
                value.copy_(tensors['basis.' + key])
        self.optimizer.load_state_dict(tensors['optimizer'])
        self.rng.setstate(_rng_state(tensors['rng']))
        if not same_parent:
            # Following a new promoted parent: the live bank continues (its basis and factors are coordinates of
            # weight updates, valid on any same-architecture parent); relative energies follow the new weights. The
            # un-meta-trained comparison bank is the same seeded construction on the new parent's weights.
            state['parents'].append({'sha256': self.state['parent_sha256'], 'from_step': state['step'],
                                     'initial_bank': self.bank0.snapshot,
                                     'scales': restore_invariants(self.bank0, self.leaves, basis=False)})
            state['parent_sha256'] = self.state['parent_sha256']
            state['initial_bank'] = self.bank0.snapshot
        self.state = state

    def save(self):
        tensors = {'step': self.state['step'], 'optimizer': self.optimizer.state_dict(), 'rng': _rng_tensor(self.rng.getstate())}
        for k, factors in self.leaves['factors_by_component'].items():
            for name, (P, Q) in factors.items():
                tensors[f'component.{k}.P.{name}'], tensors[f'component.{k}.Q.{name}'] = P.detach().clone(), Q.detach().clone()
        for key, value in self.leaves['basis'].items():
            tensors['basis.' + key] = value.detach().clone()
        partial = self.out / 'state.torch.partial'
        torch.save(tensors, partial)
        partial.replace(self.out / 'state.torch')
        record = self.out / 'state.json.partial'
        record.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        record.replace(self.out / 'state.json')

    def publish(self):
        """The live bank as an ordinary bank snapshot (`bank.save_bank`): the deployment and tuning input."""
        from dataclasses import replace
        from .bank import SharedBasis, component_id, save_bank
        restore_invariants(self.bank0, self.leaves)
        factors, basis = _detached(self.leaves)
        shared = SharedBasis(self.bank0.basis.native_identity, self.recipe.rank,
                             {name: basis['left.' + name].clone() for name in self.bank0.basis.names},
                             {name: basis['right.' + name].clone() for name in self.bank0.basis.names},
                             {**self.bank0.basis.method, 'meta_learned_from': self.bank0.basis_artifact,
                              'outer_steps': self.state['step']})
        record = shared.check().to_artifact(self.engine)
        components = [replace(c, factors={n: (P.clone(), Q.clone()) for n, (P, Q) in factors[k].items()},
                              id=component_id(c.source, factors[k]))
                      for k, c in enumerate(self.bank0.components)]
        published = replace(self.bank0.with_components(components, reason=f'meta-learning step {self.state["step"]}'),
                            basis=shared, basis_artifact=record.id)
        manifest = save_bank(self.engine, published)
        path = self.out / 'bank-manifest.json'
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        return manifest

    # -- execution ----------------------------------------------------------------------------------------------
    def _support(self, rows):
        from .fit import SupportSet
        return SupportSet(self.engine, MANIFEST, [{'binding': SCOPE, 'graph': row['graph']} for row in rows])

    def _array(self):
        from .fit import ScopeArray
        K = len(self.bank0.components)
        return ScopeArray(SCOPE, self.placeholder, list(range(K)), [True] * K)

    def _bank_values(self, which):
        if which == 'initial':
            return ({k: c.factors for k, c in enumerate(self.bank0.components)},
                    {f'{side}.{name}': getattr(self.bank0.basis, side)[name] for name in self.bank0.basis.names
                     for side in ('left', 'right')})
        return _detached(self.leaves)

    def fit(self, rows, which='meta'):
        """The inner step: coefficients only, on ``rows``, with the bank ``which`` (live or initial) fixed."""
        from .fit import fit_in_bank
        factors, basis = self._bank_values(which)
        result = fit_in_bank(self.engine, self.bank0, self._support(rows), [self._array()], steps=self.recipe.inner_steps,
                             refine_steps=self.recipe.refine_steps, learning_rate=self.recipe.inner_lr, train_states=False,
                             factors=factors, basis_leaves={self.bank0.basis_artifact: basis})
        return result

    def score(self, rows, which, executed=None, *, backward=False):
        """Mean teacher-forced loss of ``rows`` under bank ``which`` at executed coefficients (zeros: the parent).
        With ``backward`` the gradient reaches the live factors and basis (``which='meta'``)."""
        support = self._support(rows)
        G = len(self.bank0.group_labels)
        executed = torch.zeros(len(self.bank0.components), G) if executed is None else executed.detach()
        if backward:
            factors, basis = self.leaves['factors_by_component'], self.leaves['basis']
        else:
            factors, basis = self._bank_values(which)
        slots = list(range(len(self.bank0.components)))
        make = lambda: {self.placeholder: self.bank0.adapter_tensors(self.bank0.composed_cores(slots, executed, factors=factors))}
        if backward:
            return support.accumulate({SCOPE: self.placeholder}, make, extra_leaves={self.bank0.basis_artifact: basis})
        with torch.no_grad():
            return support.accumulate({SCOPE: self.placeholder}, make, backward=False,
                                      extra_leaves={self.bank0.basis_artifact: basis})

    def fit_lora(self, rows, steps):
        """The plain per-program LoRA baseline: free ``B A`` factors (B zero) trained by Adam for ``steps`` support
        passes, the same pass count as the inner fit. Returns ``(leaves, history)``."""
        record = self.engine.store.get_native_tensor_artifact(self.lora).artifact
        leaves = {key: value.detach().clone().requires_grad_(True) for key, value in record.tensors.items()}
        optimizer = torch.optim.Adam(list(leaves.values()), lr=self.recipe.lora_lr)
        support = self._support(rows)
        history = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            history.append(support.accumulate({SCOPE: self.lora}, lambda: {self.lora: leaves}))
            optimizer.step()
        return leaves, history

    def score_lora(self, rows, leaves):
        support = self._support(rows)
        with torch.no_grad():
            return support.accumulate({SCOPE: self.lora}, lambda: {self.lora: {k: v.detach() for k, v in leaves.items()}},
                                      backward=False)

    def episode(self, family):
        """Disjoint support and query worlds of one meta-training family (one row each)."""
        worlds = [w for w in self.pool['families'][family]['worlds'] if w['split'] == 'train']
        chosen = self.rng.sample(worlds, self.recipe.support_worlds + self.recipe.query_worlds)
        support = [row for w in chosen[:self.recipe.support_worlds] for row in w['rows']]
        query = [row for w in chosen[self.recipe.support_worlds:] for row in w['rows']]
        return support, query, [w['world'] for w in chosen]

    def step(self):
        """One first-order outer step over ``meta_batch`` sampled meta-training families."""
        started = time.time()
        families = sorted(name for name, f in self.pool['families'].items() if f['role'] == 'meta-train')
        if len(families) < self.recipe.meta_batch:
            raise LearnedOptimizerError('fewer meta-training families than the meta batch')
        batch = self.rng.sample(families, self.recipe.meta_batch)
        self.optimizer.zero_grad(set_to_none=True)
        records = []
        for family in batch:
            support, query, worlds = self.episode(family)
            fitted = self.fit(support)
            executed = fitted['arrays'][SCOPE]['executed']
            query_loss = self.score(query, 'meta', executed, backward=True)
            records.append({'family': family, 'worlds': worlds, 'support_start': fitted['loss']['continuous'][0],
                            'support_fitted': fitted['loss']['final'], 'query_fitted': query_loss,
                            'gain': [float(g) for g in fitted['arrays'][SCOPE]['gain']],
                            'support_components': fitted['arrays'][SCOPE]['support']})
        params = [p for group in self.optimizer.param_groups for p in group['params']]
        norms = {}
        for name, group in (('bank', self.optimizer.param_groups[0]), ('basis', self.optimizer.param_groups[1])):
            grads = [p.grad for p in group['params'] if p.grad is not None]
            for grad in grads:
                grad.div_(len(batch))
            norms[name] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else 0.
        before = [p.detach().clone() for p in params]
        self.optimizer.step()
        restore_invariants(self.bank0, self.leaves)
        update = float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(params, before))))
        self.state['step'] += 1
        entry = {'step': self.state['step'], 'parent_sha256': self.state['parent_sha256'], 'families': records,
                 'grad_norm': norms, 'update_norm': update, 'seconds': round(time.time() - started, 2),
                 'passes': len(batch) * (self.recipe.fit_passes + 1 + 1)}
        self.state['history'].append(entry)
        return entry

    def evaluate(self, split='dev', arms=ARMS, families=None):
        """Held-family adaptation: fit on each held family's train worlds, score its ``split`` worlds per arm."""
        started = time.time()
        held = [name for name, f in sorted(self.pool['families'].items()) if f['role'] == 'held']
        if families is not None:
            held = [name for name in held if name in families]
        out = {'schema': EVAL_SCHEMA, 'split': split, 'step': self.state['step'], 'parent_sha256': self.state['parent_sha256'],
               'fit_passes': self.recipe.fit_passes, 'families': {}}
        for name in held:
            worlds = self.pool['families'][name]['worlds']
            support = [row for w in worlds if w['split'] == 'train' for row in w['rows']]
            scored = [row for w in worlds if w['split'] == split for row in w['rows']]
            if not support or not scored:
                continue
            result = {}
            for arm in arms:
                if arm == 'parent':
                    result[arm] = {'loss': self.score(scored, 'initial')}
                elif arm in ('meta', 'initial'):
                    fitted = self.fit(support, arm)
                    result[arm] = {'loss': self.score(scored, arm, fitted['arrays'][SCOPE]['executed']),
                                   'support_fitted': fitted['loss']['final']}
                elif arm == 'lora':
                    leaves, history = self.fit_lora(support, self.recipe.fit_passes)
                    result[arm] = {'loss': self.score_lora(scored, leaves), 'support_last': history[-1]}
            out['families'][name] = result
        means = {}
        for arm in arms:
            values = [f[arm]['loss'] for f in out['families'].values()]
            means[arm] = sum(values) / len(values) if values else None
        out['mean'] = means
        if means.get('parent') is not None:
            out['gain_over_parent'] = {arm: means['parent'] - value for arm, value in means.items() if arm != 'parent'}
        wins = {}
        for other in ('initial', 'lora'):
            if 'meta' in arms and other in arms:
                wins[other] = sum(1 for f in out['families'].values() if f['meta']['loss'] < f[other]['loss'])
        out['meta_wins'] = wins
        out['seconds'] = round(time.time() - started, 2)
        self.state['evaluations'].append(out)
        return out


def promotion(evaluation, *, min_families=1, margin=0.):
    """The stage's held-family criterion on one evaluation: the meta-trained bank's mean held loss beats both the
    un-meta-trained bank and the plain LoRA at matched fit passes by ``margin`` (and is lower than the parent)."""
    means = evaluation.get('mean') or {}
    count = len(evaluation.get('families') or {})
    if count < min_families or any(means.get(arm) is None for arm in ARMS):
        return {'met': False, 'reason': f'{count} held families scored; all four arms are required'}
    beats = {other: means['meta'] + margin < means[other] for other in ('initial', 'lora', 'parent')}
    met = all(beats.values())
    return {'met': met, 'beats': beats, 'means': means, 'families': count,
            'reason': 'meta-trained bank beats the un-meta-trained bank, plain LoRA and the parent' if met else
            'meta-trained bank does not beat: ' + ', '.join(k for k, v in beats.items() if not v)}


def _rng_tensor(state):
    return torch.tensor(list(state[1]), dtype=torch.int64), state[0], state[2]


def _rng_state(value):
    tensor, version, gauss = value
    return version, tuple(int(x) for x in tensor.tolist()), gauss
