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
ARMS = ('parent', 'meta', 'meta_residual', 'meta_unfitted', 'initial', 'initial_unfitted', 'lora')
# Refit at every evaluation; the others are fixed per parent. ``*_unfitted``: the learned initial coefficients with no
# inner step, separating what the bank shifts for every program from what the per-program fit adds.
BANK_ARMS = ('meta', 'meta_residual', 'meta_unfitted')
PROMOTION_ARMS = ('parent', 'meta', 'initial', 'lora')
NEAR_DUPLICATE = .8                             # target-token similarity at which a held family is a near-duplicate
INNER_FIT_LAW = 'natlang.bank-inner-fit/meta-sgd-1'


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


def assistant_targets(graph):
    """Output-only supervision of a whole-trajectory source graph (AGENTS.md "Supervise the model's own outputs"):
    its declared assistant tokens keep unit weight, every context/feedback weight becomes zero. Such sources weight
    assistant tokens at >= 1 and context below it; the count must equal the conversion's declared
    ``assistant_target_tokens`` or the graph is refused (None). Graphs without that declaration keep their weights."""
    policy = (graph.get('conversion') or {}).get('supervision_policy') or {}
    declared = policy.get('assistant_target_tokens')
    if declared is None:
        return graph
    nodes, count = [], 0
    for node in graph['nodes']:
        weights = [1. if w >= 1. else 0. for w in node.get('target_weights') or ()]
        count += int(sum(weights))
        nodes.append({**node, 'target_weights': weights})
    if count != declared or not count:
        return None
    return {**graph, 'nodes': nodes, 'supervision': 'assistant-output-only-v1'}


def target_digest(graph):
    """Digest of a graph's supervised target tokens (weight > 0), node by node: two worlds with equal digests teach
    exactly the same targets. Index-only rows carry it precomputed."""
    if 'target_digest' in graph:
        return graph['target_digest']
    key = [[tok for tok, w in zip(node.get('token_ids') or (), node.get('target_weights') or ()) if w > 0]
           for node in graph.get('nodes') or ()]
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()


def graph_rows(paths, pattern, *, protected, max_tokens, prefix=None, families=None, index_only=False):
    """Rows of verified native graph sources whose id matches ``pattern`` (named groups ``family`` and ``world``).
    ``prefix``: cheap substring every candidate line contains; ``families``: keep only these; ``index_only``: rows
    carry the graph id instead of the graph (a cheap first pass over large sources). Protected-related, non-train,
    oversize and target-free graphs are skipped and counted."""
    regex = re.compile(pattern)
    counts = {'read': 0, 'matched': 0, 'protected': 0, 'oversize': 0, 'no_target': 0, 'split': 0, 'unaligned_targets': 0}
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
            graph = assistant_targets(graph)
            if graph is None:
                counts['unaligned_targets'] += 1
                continue
            rows.append({'family': match.group('family'), 'world': match.group('world'),
                         'graph': {'id': graph['id'], 'target_digest': target_digest(graph)} if index_only else graph,
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
        # Worlds whose supervised targets repeat an earlier world's exactly carry no new evidence: a support fit could
        # score a dev world by memorizing it. Keep the first of each distinct target sequence.
        seen, distinct = set(), []
        for world in sorted(families[name], key=natural):
            key = tuple(target_digest(row['graph']) for row in families[name][world])
            if key in seen:
                continue
            seen.add(key)
            distinct.append(world)
        worlds = distinct
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
    rank: int = 32                # shared basis rank R per target matrix
    components: int = 64          # bank size cap K; the PCA initialization keeps those explaining ``variance``
    component_rank: int = 16      # rank of each component core inside the basis
    variance: float = .9          # share of the meta-train families' LoRA-delta energy the initial components explain
    init_families: int = 64       # meta-training families whose LoRA fits initialize the bank (and feed the regression)
    init: str = 'pca'             # 'pca': principal directions of meta-train families' LoRA fits; 'random': seeded
    layer_groups: int = 8         # coefficient groups: depth slices x matrix family (`bank.target_groups`)
    inner_steps: int = 12         # support passes per fit, identical for every fitting arm (matched compute)
    inner_lr: float = .05         # initial per-coefficient step for the random initialization (PCA derives its own)
    residual_rank: int = 2        # per-program residual core rank, fitted alongside (its own arm); 0: none
    residual_lr: float = 1e-2
    outer_lr_bank: float = 1e-2
    outer_lr_basis: float = 1e-4
    outer_lr_meta: float = 1e-2   # learned initial coefficients and per-coefficient log step sizes (Meta-SGD)
    aux_weight: float = 1.        # weight-space regression of meta-train families' LoRA deltas (no model forward)
    aux_families: int = 8
    meta_batch: int = 2
    support_worlds: int = 3
    query_worlds: int = 2
    train_bank: bool = True       # AGENTS.md "Learned LoRA mixtures and bases": explicit
    train_basis: bool = True
    lora_rank: int = 8            # the plain per-program LoRA baseline (and the PCA initialization's fits)
    lora_lr: float = 1e-3
    reference_passes: list = field(default_factory=lambda: [1, 4, 24])  # LoRA references at these pass budgets
    target_pattern: str | None = None
    seed: int = 0

    def __post_init__(self):
        for name in ('rank', 'components', 'component_rank', 'meta_batch', 'support_worlds', 'query_worlds', 'lora_rank',
                     'inner_steps', 'layer_groups'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise LearnedOptimizerError(f'{name} must be a positive integer')
        if self.component_rank > self.rank or not 0 <= self.residual_rank <= self.rank:
            raise LearnedOptimizerError('components and the residual fit inside the basis')
        if self.init not in ('pca', 'random') or not 0 < self.variance <= 1:
            raise LearnedOptimizerError('init is pca or random; variance is a share in (0, 1]')
        if self.train_bank is not True or self.train_basis is not True:
            raise LearnedOptimizerError('bank meta-learning trains the bank and basis explicitly')

    @property
    def fit_passes(self):
        return self.inner_steps

    def as_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, value):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in value.items() if k in known})


# -- plain LoRA fits and the data-driven initialization -----------------------------------------------------------
def lora_adapter(engine, recipe, names):
    from ...model.native_adapters import create_adapter
    return create_adapter(engine, {'kind': 'free-lora', 'rank': recipe.lora_rank, 'alpha': float(recipe.lora_rank),
                                   'seed': recipe.seed, 'parameter_names': names}).id


def support_set(engine, rows):
    from .fit import SupportSet
    return SupportSet(engine, MANIFEST, [{'binding': SCOPE, 'graph': row['graph']} for row in rows])


def lora_fit(engine, adapter, rows, steps, lr):
    """Free ``B A`` factors (B zero) trained by Adam for ``steps`` support passes. Returns ``(leaves, history)``."""
    record = engine.store.get_native_tensor_artifact(adapter).artifact
    leaves = {key: value.detach().clone().requires_grad_(True) for key, value in record.tensors.items()}
    optimizer = torch.optim.Adam(list(leaves.values()), lr=lr)
    support = support_set(engine, rows)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        history.append(support.accumulate({SCOPE: adapter}, lambda: {adapter: leaves}))
        optimizer.step()
    return leaves, history


def family_support(pool, family, count):
    """The first ``count`` train worlds of a family (natural order), one row each: its LoRA-fit support."""
    worlds = [w for w in pool['families'][family]['worlds'] if w['split'] == 'train'][:count]
    return [row for w in worlds for row in w['rows']]


def family_lora_fits(engine, pool, recipe, names, directory, *, log=None):
    """Plain LoRA fits of every meta-training family on its support worlds (the baseline's parameterization and pass
    count), saved per family so an interrupted run resumes: ``{family: {name: (B, A)}}`` with ``Delta W = B A``.
    Held families never contribute."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    adapter = None
    fits = {}
    for family in sorted(name for name, f in pool['families'].items() if f['role'] == 'meta-train')[:recipe.init_families]:
        path = directory / f'{family}.pt'
        if path.exists():
            fits[family] = torch.load(path, map_location='cpu', weights_only=True)['factors']
            continue
        adapter = adapter or lora_adapter(engine, recipe, names)
        started = time.time()
        leaves, history = lora_fit(engine, adapter, family_support(pool, family, recipe.support_worlds),
                                   recipe.inner_steps, recipe.lora_lr)
        scale = 1.  # alpha == rank
        factors = {name: ((leaves['lora_B.' + name].detach() * scale).cpu().contiguous(),
                          leaves['lora_A.' + name].detach().cpu().contiguous()) for name in names}
        partial = path.with_suffix('.pt.partial')
        torch.save({'factors': factors, 'history': history, 'recipe': recipe.as_json()}, partial)
        partial.replace(path)
        fits[family] = factors
        _release_cache()
        if log is not None:
            log(family, {'support_first': round(history[0], 5), 'support_last': round(history[-1], 5),
                         'seconds': round(time.time() - started, 1)})
    return fits


def _root(gram):
    """``S`` with ``S S^T = gram`` (symmetric PSD)."""
    values, vectors = torch.linalg.eigh(gram.double())
    return (vectors * values.clamp_min(0).sqrt()[..., None, :]).float()


def delta_basis(fits, names, rank):
    """Per matrix, the top-``rank`` left and right singular directions of the stacked LoRA deltas
    ``[D_1 ... D_F]`` (computed from factors: ``D D^T = B (A A^T) B^T``), QR-exact orthonormal."""
    from .bank import orthonormalize
    left, right = {}, {}
    for name in names:
        lefts, rights = [], []
        for factors in fits.values():
            B, A = (t.float() for t in factors[name])
            lefts.append(B @ _root(A @ A.transpose(-2, -1)))
            rights.append(A.transpose(-2, -1) @ _root(B.transpose(-2, -1) @ B))
        for side, stack in ((left, lefts), (right, rights)):
            u = torch.linalg.svd(torch.cat(stack, -1).double(), full_matrices=False)[0][..., :rank].float()
            if u.shape[-1] < rank:
                raise LearnedOptimizerError(f'{name}: fewer LoRA directions than the basis rank')
            side[name] = orthonormalize(u)[0].contiguous()
    return left, right


def project_fits(fits, left, right, q):
    """Basis coordinates ``T_fm = (U^T B)(A V)`` of each fit, flattened with weights ``sqrt(q_m)`` so the Euclidean
    norm is the relative update energy; also each fit's total relative energy ``sum_m q_m ||D_fm||^2``."""
    from ..gates import factor_sq_norm
    rows, energy = [], []
    for factors in fits.values():
        parts, total = [], 0.
        for name in sorted(left):
            B, A = (t.float() for t in factors[name])
            core = (left[name].transpose(-2, -1) @ B) @ (A @ right[name])
            parts.append(math.sqrt(q[name]) * core.reshape(-1))
            total += q[name] * float(factor_sq_norm(B, A))
        rows.append(torch.cat(parts))
        energy.append(total)
    return torch.stack(rows), energy


def pca_components(bank, fits, recipe):
    """Principal directions of the fits' basis coordinates as unit-energy components (uncentered: the mean update
    is a direction too), with the initial coefficients (mean projection) and step sizes (two spreads over the inner
    steps) they imply. Returns ``(components, a0 (K,), step0 (K,), spectrum)``."""
    from .bank import Component, component_id, factor_core, normalize
    X, energy = project_fits(fits, bank.basis.left, bank.basis.right, bank.q)
    _, s, vh = torch.linalg.svd(X.double(), full_matrices=False)
    share = (s ** 2) / float((s ** 2).sum())
    cumulative = torch.cumsum(share, 0)
    count = min(recipe.components, len(s), int((cumulative < recipe.variance).sum()) + 1)
    sizes = [(name, bank.basis.left[name].shape[:-2], bank.basis.rank) for name in sorted(bank.basis.left)]
    components, a0, step0 = [], [], []
    for k in range(count):
        direction, offset, factors = vh[k].float(), 0, {}
        for name, lead, R in sizes:
            n = int(torch.Size(lead).numel()) * R * R
            core = (direction[offset:offset + n] / math.sqrt(bank.q[name])).reshape(*lead, R, R)
            offset += n
            factors[name] = factor_core(core, recipe.component_rank)
        source = {'kind': 'lora-pca', 'index': k, 'share': float(share[k]), 'families': len(fits)}
        component, scale = normalize(Component(component_id(source, factors), factors, None, f'LoRA principal direction {k}',
                                               source, ('tree', 'folder', 'lambda')), bank.q)
        coefficients = (X.double() @ vh[k]).float() / scale
        components.append(component)
        a0.append(float(coefficients.mean()))
        step0.append(max(2 * float(coefficients.std()) / recipe.inner_steps, 1e-6))
    captured = float(X.pow(2).sum()) / max(sum(energy), 1e-30)
    spectrum = {'families': len(fits), 'components': count, 'share': [round(float(v), 5) for v in share[:max(count, 16)]],
                'explained': round(float(cumulative[count - 1]), 5), 'basis_captures': round(captured, 5)}
    return components, torch.tensor(a0), torch.tensor(step0), spectrum


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


def meta_sgd(a0, log_step, steps, support_pass, *, after_step=None):
    """The published inner-fit law (`INNER_FIT_LAW`): ``steps`` Adam-preconditioned steps on free per-group
    coefficients from ``a0``, each scaled by ``exp(log_step)``. ``support_pass(a)`` runs the support loss with its
    backward into ``a`` and returns the loss value. Returns the fitted coefficients (detached), the summed step
    directions and the per-step support losses. Coefficients a pass masks out get a zero gradient and stay put."""
    step = log_step.exp()
    a = a0.detach().clone().requires_grad_(True)
    first, second, travel = torch.zeros_like(a), torch.zeros_like(a), torch.zeros_like(a)
    history = []
    for t in range(1, steps + 1):
        a.grad = None
        history.append(support_pass(a))
        grad = a.grad if a.grad is not None else torch.zeros_like(a)
        with torch.no_grad():
            first.mul_(.9).add_(grad, alpha=.1)
            second.mul_(.999).addcmul_(grad, grad, value=.001)
            direction = (first / (1 - .9 ** t)) / ((second / (1 - .999 ** t)).sqrt() + 1e-8)
            a.sub_(step * direction)
            travel.add_(direction)
        if after_step is not None:
            after_step()
    return a.detach(), travel, history


class _TrainAll:
    train_bank = True
    train_basis = True



class BankMetaLearner:
    """One operation's bank, live leaves, meta-parameters, outer optimizer and history (``out/state.json`` +
    ``out/state.torch``).

    The inner fit (`_inner`) is Meta-SGD on free signed per-group coefficients ``a (K, G)``: it starts at the learned
    initial coefficients ``a0`` and takes ``inner_steps`` Adam-preconditioned steps scaled by learned per-coefficient
    step sizes ``exp(log_step)``. Executed cores are ``C_m = sum_k a[k, g(m)] P_k Q_k^T`` (+ an optional per-program
    residual ``X_m Y_m^T`` in basis coordinates). First-order outer gradients: the query loss at the fitted
    coefficients reaches the factors and basis directly, ``a0`` through ``dL/da`` and the step sizes through
    ``dL/da * (-travel)`` (each step's direction summed)."""

    def __init__(self, engine, pool, recipe, out, *, parent_sha256, fits=None):
        from .bank import new_bank, orthonormalize, weight_svd_basis, SharedBasis
        from .fit import placeholder_adapters
        from .mixture import MixtureFormat
        from ...model.native_adapters import target_parameters
        self.engine, self.pool, self.recipe, self.out = engine, pool, recipe, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.names = select_targets(engine, recipe)
        names = self.names
        torch.manual_seed(recipe.seed)
        if recipe.init == 'pca':
            if not fits:
                raise LearnedOptimizerError('the PCA initialization needs the meta-training families\' LoRA fits')
            left, right = delta_basis(fits, names, recipe.rank)
            basis = SharedBasis(dict(engine.native_identity), recipe.rank, left, right,
                                {'kind': 'lora-delta-svd', 'sources': [], 'families': len(fits)})
        else:
            basis = weight_svd_basis(engine, names, recipe.rank)
            # Device SVDs (cuSOLVER) return columns orthonormal only to ~1e-4; the bank's exact invariants need them
            # to FP32 precision: re-orthonormalize by QR (same span, sign-canonical R).
            for side in (basis.left, basis.right):
                for name in names:
                    side[name] = orthonormalize(side[name])[0].contiguous()
        bank = new_bank(engine, basis, MixtureFormat(), quarters=min(recipe.layer_groups, engine.shared_model.port.num_layers))
        G = len(bank.group_labels)
        if recipe.init == 'pca':
            components, a0, step0, self.spectrum = pca_components(bank, fits, recipe)
        else:
            components = random_components(bank, recipe.components, recipe.component_rank, recipe.seed)
            a0, step0, self.spectrum = torch.zeros(len(components)), torch.full((len(components),), recipe.inner_lr), None
        self.bank0 = bank.with_components(components, reason=f'{recipe.init} initialization')
        self.a0 = a0[:, None].expand(-1, G).contiguous()
        self.log_step0 = step0.log()[:, None].expand(-1, G).contiguous()
        self.fits = fits or {}
        self.placeholder = placeholder_adapters(engine, self.bank0, [SCOPE])[SCOPE]
        self.lora = lora_adapter(engine, recipe, names)
        self.leaves = _snapshot_leaves(self.bank0)
        self.meta = {'init': self.a0.clone().requires_grad_(True), 'log_step': self.log_step0.clone().requires_grad_(True)}
        self.optimizer = torch.optim.Adam([
            {'params': [t for f in self.leaves['factors_by_component'].values() for pair in f.values() for t in pair],
             'lr': recipe.outer_lr_bank},
            {'params': list(self.leaves['basis'].values()), 'lr': recipe.outer_lr_basis},
            {'params': [self.meta['init'], self.meta['log_step']], 'lr': recipe.outer_lr_meta}])
        self.state = {'schema': STATE_SCHEMA, 'recipe': recipe.as_json(), 'parent_sha256': parent_sha256,
                      'parents': [{'sha256': parent_sha256, 'from_step': 0}], 'spectrum': self.spectrum,
                      'initial_bank': self.bank0.snapshot, 'targets': len(names), 'groups': G,
                      'targets_below_rank': excluded_targets(engine, recipe),
                      'components': len(components), 'step': 0, 'history': [], 'evaluations': [], 'rng': recipe.seed}
        self.rng = random.Random(recipe.seed)
        self._restore()

    # -- persistence --------------------------------------------------------------------------------------------
    def _restore(self):
        path = self.out / 'state.json'
        if not path.exists():
            return
        state = json.loads(path.read_text())
        # Fields added later read their default from an older saved recipe.
        if state.get('schema') != STATE_SCHEMA or MetaRecipe.from_json(state['recipe']).as_json() != self.recipe.as_json():
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
            for key, value in self.meta.items():
                value.copy_(tensors['meta.' + key])
        self.optimizer.load_state_dict(tensors['optimizer'])
        self.rng.setstate(_rng_state(tensors['rng']))
        if not same_parent:
            # Following a new promoted parent: the live bank continues (its basis and factors are coordinates of
            # weight updates, valid on any same-architecture parent); relative energies follow the new weights. The
            # un-meta-trained comparison bank is the same construction on the new parent's weights.
            scales = restore_invariants(self.bank0, self.leaves, basis=False)
            self._rescale_meta(scales)
            state['parents'].append({'sha256': self.state['parent_sha256'], 'from_step': state['step'],
                                     'initial_bank': self.bank0.snapshot, 'scales': scales})
            state['parent_sha256'] = self.state['parent_sha256']
            state['initial_bank'] = self.bank0.snapshot
        self.state = state

    def _rescale_meta(self, scales):
        """Components rescaled by ``s_k`` keep every composed update when their coefficients and steps scale by
        ``1/s_k``."""
        with torch.no_grad():
            scale = torch.tensor(scales, dtype=torch.float32)[:, None]
            self.meta['init'].div_(scale)
            self.meta['log_step'].sub_(scale.log())

    def save(self):
        tensors = {'step': self.state['step'], 'optimizer': self.optimizer.state_dict(), 'rng': _rng_tensor(self.rng.getstate())}
        for k, factors in self.leaves['factors_by_component'].items():
            for name, (P, Q) in factors.items():
                tensors[f'component.{k}.P.{name}'], tensors[f'component.{k}.Q.{name}'] = P.detach().clone(), Q.detach().clone()
        for key, value in self.leaves['basis'].items():
            tensors['basis.' + key] = value.detach().clone()
        for key, value in self.meta.items():
            tensors['meta.' + key] = value.detach().clone()
        partial = self.out / 'state.torch.partial'
        torch.save(tensors, partial)
        partial.replace(self.out / 'state.torch')
        record = self.out / 'state.json.partial'
        record.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        record.replace(self.out / 'state.json')

    def snapshot_bank(self):
        """The live bank as a stored bank (invariants restored, basis artifact written): the form adapters of the
        candidate are composed on. Cached per outer step."""
        from dataclasses import replace
        from .bank import SharedBasis, component_id
        cached = getattr(self, '_snapshot', None)
        if cached is not None and cached[0] == self.state['step']:
            return cached[1]
        self._rescale_meta(restore_invariants(self.bank0, self.leaves))
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
        bank = replace(self.bank0.with_components(components, reason=f'meta-learning step {self.state["step"]}'),
                       basis=shared, basis_artifact=record.id)
        self._snapshot = (self.state['step'], bank)
        return bank

    def publish(self):
        """The live bank as an ordinary bank snapshot (`bank.save_bank`) plus its inner-fit law (initial
        coefficients and step sizes, a parameter-set artifact): the deployment and tuning input."""
        from ...model.native_artifact import NativeTensorArtifact
        from .bank import save_bank
        bank = self.snapshot_bank()
        manifest = save_bank(self.engine, bank)
        law = self.engine.store.put_native_tensor_artifact(NativeTensorArtifact(
            'parameter-set', self.engine.native_identity,
            {key: value.detach().cpu().contiguous() for key, value in self.meta.items()}, bank.basis_artifact,
            {'scope': INNER_FIT_LAW, 'snapshot': manifest['snapshot']}))
        manifest['inner_fit'] = {'law': INNER_FIT_LAW, 'tensors': law.id, 'steps': self.recipe.inner_steps,
                                 'residual_rank': self.recipe.residual_rank, 'residual_lr': self.recipe.residual_lr}
        path = self.out / 'bank-manifest.json'
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        return manifest

    def promote(self, manifest):
        """The candidate passed the task-outcome gate: it becomes the published bank later candidates must beat."""
        self.state['published'] = {'manifest': manifest, 'step': self.state['step']}
        self._published = None

    def _published_values(self):
        """``(bank, init, log_step)`` of the published bank; before any promotion, the un-meta-trained bank."""
        record = self.state.get('published')
        if record is None:
            return self.bank0, self.a0, self.log_step0
        if getattr(self, '_published', None) is None:
            from .bank import load_bank
            manifest = record['manifest']
            law = self.engine.store.get_native_tensor_artifact(manifest['inner_fit']['tensors']).artifact.tensors
            self._published = (load_bank(self.engine, manifest), law['init'], law['log_step'])
        return self._published

    def bank_of(self, which):
        if which == 'initial':
            return self.bank0
        if which == 'published':
            return self._published_values()[0]
        return self.snapshot_bank()

    # -- execution ----------------------------------------------------------------------------------------------
    def _values(self, which):
        """Fixed bank values and meta-parameters for an inner fit: the live (``meta``), initial or published bank."""
        if which == 'published':
            bank, init, log_step = self._published_values()
            return ({k: c.factors for k, c in enumerate(bank.components)},
                    {f'{side}.{name}': getattr(bank.basis, side)[name] for name in bank.basis.names
                     for side in ('left', 'right')}, init, log_step)
        if which == 'initial':
            return ({k: c.factors for k, c in enumerate(self.bank0.components)},
                    {f'{side}.{name}': getattr(self.bank0.basis, side)[name] for name in self.bank0.basis.names
                     for side in ('left', 'right')}, self.a0, self.log_step0)
        factors, basis = _detached(self.leaves)
        return factors, basis, self.meta['init'].detach(), self.meta['log_step'].detach()

    def _tensors(self, coefficients, factors, residual=None):
        slots = list(range(len(self.bank0.components)))
        cores = self.bank0.composed_cores(slots, coefficients, factors=factors)
        if residual:
            cores = {name: core + residual[name][0] @ residual[name][1].transpose(-2, -1) if name in residual else core
                     for name, core in cores.items()}
        return {self.placeholder: self.bank0.adapter_tensors(cores)}

    def _residual(self):
        r = self.recipe.residual_rank
        generator = torch.Generator().manual_seed(self.recipe.seed)
        out = {}
        for name in self.bank0.basis.names:
            lead, R = self.bank0.basis.left[name].shape[:-2], self.recipe.rank
            out[name] = (torch.zeros(*lead, R, r).requires_grad_(True),
                         (torch.randn(*lead, R, r, generator=generator) / math.sqrt(R)).requires_grad_(True))
        return out

    def fit(self, rows, which='meta', *, residual=False):
        """The inner step on ``rows`` with bank ``which`` held fixed: Meta-SGD on the coefficients (and, with
        ``residual``, Adam on a per-program residual core of rank ``residual_rank``). Returns the fitted
        coefficients, residual, summed step directions and the support-loss history."""
        factors, basis, a0, log_step = self._values(which)
        support = support_set(self.engine, rows)
        res = self._residual() if residual and self.recipe.residual_rank else None
        optimizer = torch.optim.Adam([t for pair in res.values() for t in pair], lr=self.recipe.residual_lr) if res else None
        extra = {self.bank0.basis_artifact: basis}

        def support_pass(a):
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            loss = support.accumulate({SCOPE: self.placeholder}, lambda: self._tensors(a, factors, res), extra_leaves=extra)
            return loss

        a, travel, history = meta_sgd(a0, log_step, self.recipe.inner_steps, support_pass,
                                      after_step=optimizer.step if optimizer else None)
        return {'coefficients': a, 'travel': travel, 'history': history,
                'residual': {n: (X.detach(), Y.detach()) for n, (X, Y) in res.items()} if res else None}

    def score(self, rows, which, coefficients=None, residual=None):
        """Mean teacher-forced loss of ``rows`` under bank ``which`` at ``coefficients`` (None: the parent)."""
        factors, basis, _, _ = self._values(which)
        support = support_set(self.engine, rows)
        G = len(self.bank0.group_labels)
        coefficients = torch.zeros(len(self.bank0.components), G) if coefficients is None else coefficients
        with torch.no_grad():
            return support.accumulate({SCOPE: self.placeholder}, lambda: self._tensors(coefficients, factors, residual),
                                      backward=False, extra_leaves={self.bank0.basis_artifact: basis})

    def query_backward(self, rows, coefficients):
        """Query loss at the fitted coefficients with gradients into the live factors and basis; returns the loss
        and ``dL/da``."""
        support = support_set(self.engine, rows)
        a = coefficients.detach().clone().requires_grad_(True)
        loss = support.accumulate({SCOPE: self.placeholder},
                                  lambda: self._tensors(a, self.leaves['factors_by_component']),
                                  extra_leaves={self.bank0.basis_artifact: self.leaves['basis']})
        return loss, (a.grad if a.grad is not None else torch.zeros_like(a))

    def fit_lora(self, rows, steps):
        return lora_fit(self.engine, self.lora, rows, steps, self.recipe.lora_lr)

    def score_lora(self, rows, leaves):
        support = support_set(self.engine, rows)
        with torch.no_grad():
            return support.accumulate({SCOPE: self.lora}, lambda: {self.lora: {k: v.detach() for k, v in leaves.items()}},
                                      backward=False)

    def regression(self, families):
        """Weight-space auxiliary loss (no model forward): per meta-train family, the share of its LoRA delta's
        relative energy that the live bank's best per-group coefficient combination leaves unexplained,
        ``(sum_m q_m ||U C(a*) V^T - D_m||^2) / sum_m q_m ||D_m||^2``. ``a*`` is the least-squares solution (held
        fixed: envelope gradient); the gradient reaches components and basis. Returns the mean (differentiable)."""
        from ..gates import factor_sq_norm
        bank, K, G = self.bank0, len(self.bank0.components), len(self.bank0.group_labels)
        factors, basis = self.leaves['factors_by_component'], self.leaves['basis']
        cores = {name: torch.stack([factors[k][name][0] @ factors[k][name][1].transpose(-2, -1) for k in range(K)])
                 for name in bank.basis.names}
        total = 0.
        for family in families:
            fit = self.fits[family]
            targets, energy = {}, 0.
            for name in bank.basis.names:
                B, A = (t.float() for t in fit[name])
                targets[name] = (basis['left.' + name].transpose(-2, -1) @ B) @ (A @ basis['right.' + name])
                energy += bank.q[name] * float(factor_sq_norm(B, A))
            a = torch.zeros(K, G)
            with torch.no_grad():
                for g in range(G):
                    members = [n for n in bank.basis.names if bank.group_of[n] == g]
                    phi = torch.cat([math.sqrt(bank.q[n]) * cores[n].reshape(K, -1) for n in members], 1).double()
                    rhs = torch.cat([math.sqrt(bank.q[n]) * targets[n].reshape(-1) for n in members]).double()
                    gram = phi @ phi.T
                    a[:, g] = torch.linalg.solve(gram + 1e-8 * float(gram.diagonal().mean()) * torch.eye(K, dtype=gram.dtype),
                                                 phi @ rhs).float()
            loss = energy
            for name in bank.basis.names:
                composed = torch.einsum('k,k...->...', a[:, bank.group_of[name]], cores[name])
                loss = loss + bank.q[name] * ((composed - targets[name]).pow(2).sum() - targets[name].pow(2).sum())
            total = total + loss / max(energy, 1e-30)
        return total / len(families)

    def episode(self, family):
        """Disjoint support and query worlds of one meta-training family (one row each)."""
        worlds = [w for w in self.pool['families'][family]['worlds'] if w['split'] == 'train']
        chosen = self.rng.sample(worlds, self.recipe.support_worlds + self.recipe.query_worlds)
        support = [row for w in chosen[:self.recipe.support_worlds] for row in w['rows']]
        query = [row for w in chosen[self.recipe.support_worlds:] for row in w['rows']]
        return support, query, [w['world'] for w in chosen]

    def step(self):
        """One first-order outer step over ``meta_batch`` sampled meta-training families, plus the weight-space
        regression over ``aux_families`` families with LoRA fits."""
        started = time.time()
        families = sorted(name for name, f in self.pool['families'].items() if f['role'] == 'meta-train')
        if len(families) < self.recipe.meta_batch:
            raise LearnedOptimizerError('fewer meta-training families than the meta batch')
        batch = self.rng.sample(families, self.recipe.meta_batch)
        self.optimizer.zero_grad(set_to_none=True)
        records, init_grad, step_grad = [], torch.zeros_like(self.a0), torch.zeros_like(self.a0)
        for family in batch:
            support, query, worlds = self.episode(family)
            fitted = self.fit(support)
            query_loss, grad = self.query_backward(query, fitted['coefficients'])
            init_grad += grad / len(batch)
            step_grad += grad * -fitted['travel'] * self.meta['log_step'].detach().exp() / len(batch)
            records.append({'family': family, 'worlds': worlds, 'support_start': fitted['history'][0],
                            'support_fitted': fitted['history'][-1], 'query_fitted': query_loss,
                            'coefficient_norm': float(fitted['coefficients'].norm())})
        for group in self.optimizer.param_groups[:2]:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.div_(len(batch))
        regression = None
        aux = sorted(self.fits)
        if self.recipe.aux_weight and aux:
            chosen = self.rng.sample(aux, min(self.recipe.aux_families, len(aux)))
            value = self.regression(chosen)
            (self.recipe.aux_weight * value).backward()
            regression = float(value.detach())
        self.meta['init'].grad, self.meta['log_step'].grad = init_grad, step_grad
        params = [p for group in self.optimizer.param_groups for p in group['params']]
        norms = {}
        for name, group in zip(('bank', 'basis', 'meta'), self.optimizer.param_groups):
            grads = [p.grad for p in group['params'] if p.grad is not None]
            norms[name] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else 0.
        before = [p.detach().clone() for p in params]
        self.optimizer.step()
        self._rescale_meta(restore_invariants(self.bank0, self.leaves))
        update = float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(params, before))))
        self.state['step'] += 1
        _release_cache()
        entry = {'step': self.state['step'], 'parent_sha256': self.state['parent_sha256'], 'families': records,
                 'regression_unexplained': regression, 'grad_norm': norms, 'update_norm': update,
                 'step_size_mean': float(self.meta['log_step'].detach().exp().mean()),
                 'seconds': round(time.time() - started, 2),
                 'passes': len(batch) * (self.recipe.inner_steps + 1)}
        self.state['history'].append(entry)
        return entry

    def evaluate(self, split='dev', arms=None, families=None, log=None):
        """Held-family adaptation: fit on each held family's train worlds, score its ``split`` worlds per arm.
        Each family also records ``similarity``: the highest target-token similarity of a scored world to one of
        its train worlds (near-duplicates make every fit look good; means are reported with and without them)."""
        arms = arms or self.arms
        started = time.time()
        held = [name for name, f in sorted(self.pool['families'].items()) if f['role'] == 'held']
        if families is not None:
            held = [name for name in held if name in families]
        out = {'schema': EVAL_SCHEMA, 'split': split, 'step': self.state['step'], 'parent_sha256': self.state['parent_sha256'],
               'fit_passes': self.recipe.fit_passes, 'families': {}}
        # Resumable: each finished family is saved, so a preempted evaluation continues where it stopped. Arms that do
        # not depend on the meta-trained bank (parent, initial bank, plain LoRA) are measured once per parent and
        # reused at later steps: same rows, same deterministic fits.
        partial = self.state.setdefault('partial_evaluations', {}).setdefault(f'{split}:{self.state["step"]}', {})
        fixed = self.state.setdefault('fixed_arms', {}).setdefault(f'{self.state["parent_sha256"]}:{split}', {})
        for name in held:
            worlds = self.pool['families'][name]['worlds']
            support = [row for w in worlds if w['split'] == 'train' for row in w['rows']]
            scored = [row for w in worlds if w['split'] == split for row in w['rows']]
            if not support or not scored:
                continue
            if name in partial:
                out['families'][name] = partial[name]
                continue
            result = {'similarity': similarity(support, scored)}
            for arm in arms:
                if arm not in BANK_ARMS and arm in fixed.get(name, {}):
                    result[arm] = fixed[name][arm]
                elif arm == 'parent':
                    result[arm] = {'loss': self.score(scored, 'initial')}
                elif arm == 'meta' and self.state['step'] == 0 and self.state.get('published') is None:
                    # Before any outer step the live bank, initial coefficients and step sizes are the initial
                    # bank's exactly: one fit serves both arms.
                    if 'initial' not in fixed.get(name, {}):
                        fitted = self.fit(support, 'initial')
                        fixed.setdefault(name, {})['initial'] = {
                            'loss': self.score(scored, 'initial', fitted['coefficients']),
                            'support_first': fitted['history'][0], 'support_last': fitted['history'][-1]}
                    result[arm] = dict(fixed[name]['initial'])
                    result['initial'] = fixed[name]['initial']
                elif arm == 'initial' and arm in result:
                    pass
                elif arm in ('meta_unfitted', 'initial_unfitted'):
                    which = arm.split('_')[0]
                    result[arm] = {'loss': self.score(scored, which, self._values(which)[2])}
                elif arm in ('meta', 'meta_residual', 'initial'):
                    which = 'initial' if arm == 'initial' else 'meta'
                    fitted = self.fit(support, which, residual=arm == 'meta_residual')
                    result[arm] = {'loss': self.score(scored, which, fitted['coefficients'], fitted['residual']),
                                   'support_first': fitted['history'][0], 'support_last': fitted['history'][-1]}
                elif arm == 'lora' or arm.startswith('lora@'):
                    passes = self.recipe.fit_passes if arm == 'lora' else int(arm.split('@')[1])
                    leaves, history = self.fit_lora(support, passes)
                    result[arm] = {'loss': self.score_lora(scored, leaves), 'support_last': history[-1]}
                if arm not in BANK_ARMS:
                    fixed.setdefault(name, {})[arm] = result[arm]
                _release_cache()
            out['families'][name] = partial[name] = result
            self.save()
            if log is not None:
                log(name, {'similarity': round(result['similarity'], 3),
                           **{arm: round(result[arm]['loss'], 5) for arm in arms}})
        out.update(summarize(out['families'], arms))
        out['seconds'] = round(time.time() - started, 2)
        self.state['partial_evaluations'].pop(f'{split}:{self.state["step"]}', None)
        self.state['evaluations'].append(out)
        return out

    @property
    def arms(self):
        """The steering arms: ``lora`` is the matched-pass reference, ``lora@n`` the other reference budgets."""
        base = ARMS if self.recipe.residual_rank else tuple(arm for arm in ARMS if arm != 'meta_residual')
        return (*base, *[f'lora@{n}' for n in self.recipe.reference_passes if n != self.recipe.fit_passes])


def similarity(support, scored):
    """Highest `difflib` ratio between a scored row's target tokens and a support row's."""
    from difflib import SequenceMatcher
    def tokens(row):
        return [tok for node in row['graph']['nodes']
                for tok, w in zip(node.get('token_ids') or (), node.get('target_weights') or ()) if w > 0]
    best = 0.
    for left in map(tokens, scored):
        for right in map(tokens, support):
            best = max(best, SequenceMatcher(None, left, right, autojunk=False).ratio())
    return best


def summarize(families, arms, *, distinct_below=NEAR_DUPLICATE):
    """Means, gains over the parent and meta wins over all scored families and over the families without a
    near-duplicate scored world (``similarity < distinct_below``)."""
    out = {}
    for label, chosen in (('', families), ('_distinct', {k: v for k, v in families.items()
                                                          if v.get('similarity', 0.) < distinct_below})):
        means = {}
        for arm in arms:
            values = [f[arm]['loss'] for f in chosen.values() if arm in f]
            means[arm] = sum(values) / len(values) if values else None
        out['mean' + label] = means
        out['families' + label + '_count'] = len(chosen)
        if means.get('parent') is not None:
            out['gain_over_parent' + label] = {arm: means['parent'] - value for arm, value in means.items()
                                               if arm != 'parent' and value is not None}
        out['meta_wins' + label] = {other: sum(1 for f in chosen.values() if f['meta']['loss'] < f[other]['loss'])
                                    for other in ('initial', 'lora') if 'meta' in arms and other in arms}
    return out


def promotion(evaluation, *, min_families=1, margin=0.):
    """The stage's held-family criterion on one evaluation: the meta-trained bank's coefficient fit (``meta``) has a
    mean held loss below the un-meta-trained bank, the plain LoRA at matched fit passes and the parent, by
    ``margin``."""
    means = evaluation.get('mean') or {}
    count = len(evaluation.get('families') or {})
    if count < min_families or any(means.get(arm) is None for arm in PROMOTION_ARMS):
        return {'met': False, 'reason': f'{count} held families scored; the parent, meta, initial and lora arms are required'}
    beats = {other: means['meta'] + margin < means[other] for other in ('initial', 'lora', 'parent')}
    met = all(beats.values())
    return {'met': met, 'beats': beats, 'means': means, 'families': count,
            'reason': 'meta-trained bank beats the un-meta-trained bank, plain LoRA and the parent' if met else
            'meta-trained bank does not beat: ' + ', '.join(k for k, v in beats.items() if not v)}


def excluded_targets(engine, recipe):
    """Pattern-selected matrices left out because a side is shorter than the basis rank."""
    from ...model.native_adapters import target_parameters
    available = target_parameters(engine.shared_model)
    return sorted(name for name in available if (not recipe.target_pattern or re.search(recipe.target_pattern, name))
                  and min(available[name].shape[-2:]) < recipe.rank)


def select_targets(engine, recipe):
    from ...model.native_adapters import target_parameters
    available = target_parameters(engine.shared_model)
    names = sorted(available)
    if recipe.target_pattern:
        names = [name for name in names if re.search(recipe.target_pattern, name)]
    # One basis rank serves every matrix: a matrix with a side shorter than the rank (LFM3's 16-row per-head
    # attention gates at rank 32) cannot hold an orthonormal rank-R basis and is left to the parent.
    names = [name for name in names if min(available[name].shape[-2:]) >= recipe.rank]
    if not names:
        raise LearnedOptimizerError('no target matrices selected')
    return names


def _release_cache():
    """Return cached device blocks between families and steps: rows of different lengths fragment the caching
    allocator, and on unified-memory hosts the cache counts against the job's memory claim."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rng_tensor(state):
    return torch.tensor(list(state[1]), dtype=torch.int64), state[0], state[2]


def _rng_state(value):
    tensor, version, gauss = value
    return version, tuple(int(x) for x in tensor.tolist()), gauss
