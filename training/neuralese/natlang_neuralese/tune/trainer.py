"""Joint path-routed program adapter training (plans/HIERARCHICAL_LORA_TUNING.md §5.1, §5.3, §5.4).

One logical update trains every tree/folder/lambda node of a program and its per-binding family gates together.
Every objective -- supervised rows, RL episodes and RFT episodes -- runs one whole graph through the production native
gradient service (`serve.native_grad.native_graph_gradient`, the same one-forward graph objective production training
uses): each invocation under its binding's own stack, the current node coefficients, gates and soft states substituted
as differentiable leaves, and gradients through every producer/consumer dependency of the graph, Neuralese included.
A supervised row may hold several invocations (``bindings``: node name -> binding). All rows of a batch accumulate gradients against one
immutable snapshot; only then does each active node take one optimizer step through `optimize_adapter`, which keeps
the immutable coefficient lineage and optimizer continuation contract. Published family tensors stay frozen: a gate
scales its family's frozen tensors, so the gate alone receives the gradient.

Native-subspace nodes slice one shared canonical maximum-rank basis per target matrix, so a stack sums exactly in that
basis. Free-LoRA nodes carry their own factors.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, field

import torch

from ..common.hashing import canonical_json_sha256_hex
from ..data.native_graph_schema import parse_graph
from ..model.native_adapters import (create_adapter, create_canonical_basis, optimize_adapter, resolve_adapters,
                                     target_parameters)
from ..model.native_artifact import NativeTensorArtifact
from . import TuneMathError
from .rl import call_of
from .gates import factor_sq_norm, gate_l1, project_gates
from .targets import module_map

STATE_SCHEMA = 'natlang.tune.trainer-state/1'
LEVELS = ('tree', 'folder', 'lambda')


@dataclass
class Recipe:
    """One program fit's settings.

    Program tuning is a deliberate narrow specialization on few roots: small, freshly initialized parameters, few
    steps, stopped on development task success. The defaults are starting points chosen for that regime and are
    selected per program on development data by the sweep (§5.6), never on held data:

    - Coefficient cores/factors: Adam at 2e-3, about ten times a full-weight fine-tune rate, because they start at zero
      in a low-rank subspace and must move within a short schedule.
    - Per-level scales: lambda 1, folder 0.5, tree 0.25. Shared ancestors receive every descendant's gradient and
      serve many calls; a gentler rate keeps them from absorbing one lambda's specifics.
    - Soft instruction/skill states: their own Adam at 1e-3. They are activation-valued KV/hybrid tensors initialized
      from text; Adam's per-element steps are already scale-free, and a lower rate keeps them near their text meaning.
    - Gates: 1e-2 within [0, a_max].
    - Schedule: linear warmup over 5% of the steps, cosine decay to 10%, applied identically to every group's base
      rate; the position is the persisted step, so a resumed run continues the same curve. Changing the step budget
      changes the remaining curve.
    - `epochs` (passes over the admitted rows of all objectives) can replace `steps`.
    - `objectives`: logical step k trains objective objectives[k % n] (each with its own cursor), so ["sft", "rl"]
      alternates supervised and policy updates of the same state; one step never mixes objectives. "sft" supervises
      the teacher's turns; "rl" is the clipped group-relative policy objective over whole recorded episodes; "rft" is
      global supervision of whole successful episodes (quality >= `rft_min_quality`): teacher-forced loss on every
      action of the episode's complete graph, backpropagated through every producer/consumer dependency (Neuralese
      spans and soft states included) into every node and state that ran. An objective without admitted rows or
      episodes is left out of the cycle and reported.
    - `basis`: a learned optimizer bank basis artifact (tune/learned/bank.py) for native-subspace nodes instead of the
      weight-SVD canonical basis. Nodes then slice that shared basis, so a fit's residuals are directly coordinates of
      the learned optimizer's bank: its free tunes are exact targets and its releases are exact warm starts.
    """
    steps: int = 1
    epochs: float | None = None
    batch_rows: int = 8
    # Auxiliary prompt supervision is adjustable independently of the reply target.
    prompt_weight: float = .1
    seed: int = 0
    parameterization: str = 'native-subspace'
    ranks: dict = field(default_factory=lambda: {'tree': 8, 'folder': 4, 'lambda': 2})
    depth_penalty: dict = field(default_factory=lambda: {'tree': 1e-4, 'folder': 1e-4, 'lambda': 1e-4})
    tau: float = 1e-3
    a_max: float = 1.0
    optimizer: dict = field(default_factory=lambda: {'name': 'adam', 'learning_rate': 2e-3})
    level_lr_scale: dict = field(default_factory=lambda: {'tree': .25, 'folder': .5, 'lambda': 1.})
    gate_learning_rate: float = 1e-2
    target_classes: list | None = None
    free_lora_alpha: float | None = None
    # Soft instruction/skill states (plan §3.3): the shared native-state optimizer's settings.
    state_optimizer: dict = field(default_factory=lambda: {'name': 'adam', 'learning_rate': 1e-3})
    schedule: dict = field(default_factory=lambda: {'kind': 'warmup-cosine', 'warmup_ratio': .05, 'min_ratio': .1})
    objectives: list = field(default_factory=lambda: ['sft'])
    basis: str | None = None
    rft_min_quality: float = 1.

    @classmethod
    def from_json(cls, value):
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise TuneMathError(f'unknown recipe fields {sorted(unknown)}')
        recipe = cls(**value)
        if (type(recipe.prompt_weight) not in (int, float) or not math.isfinite(recipe.prompt_weight)
                or not 0 <= recipe.prompt_weight <= 1):
            raise TuneMathError('prompt_weight must be a finite number in [0, 1]')
        if recipe.parameterization not in ('native-subspace', 'free-lora'):
            raise TuneMathError('parameterization must be native-subspace or free-lora')
        if recipe.basis is not None and (recipe.parameterization != 'native-subspace' or not isinstance(recipe.basis, str)):
            raise TuneMathError('a declared basis is a bank basis artifact ID of a native-subspace fit')
        if set(recipe.ranks) - set(LEVELS) or any(type(rank) is not int or rank < 0 for rank in recipe.ranks.values()):
            raise TuneMathError('ranks map scope levels to nonnegative integers (0: no adapter at that level)')
        if any(not math.isfinite(recipe.depth_penalty.get(level, math.nan)) or recipe.depth_penalty[level] < 0
               for level, rank in recipe.ranks.items() if rank):
            raise TuneMathError('every trained level needs a finite nonnegative depth penalty')
        if type(recipe.steps) is not int or recipe.steps < 0 or type(recipe.batch_rows) is not int or recipe.batch_rows < 1:
            raise TuneMathError('steps and batch_rows must be positive integers')
        if recipe.epochs is not None and (type(recipe.epochs) not in (int, float) or not math.isfinite(recipe.epochs) or recipe.epochs <= 0):
            raise TuneMathError('epochs must be a positive number')
        if (set(recipe.level_lr_scale) != set(LEVELS) or any(type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 for value in recipe.level_lr_scale.values())):
            raise TuneMathError('level_lr_scale maps tree, folder and lambda to finite nonnegative scales')
        lr_factor(recipe.schedule, 0, 1)  # validates the schedule
        if (not isinstance(recipe.objectives, list) or not recipe.objectives
                or any(item not in ('sft', 'rl', 'rft') for item in recipe.objectives)):
            raise TuneMathError('objectives is a nonempty list of sft, rl and rft')
        if type(recipe.rft_min_quality) not in (int, float) or not 0 < recipe.rft_min_quality <= 1:
            raise TuneMathError('rft_min_quality is a quality in (0, 1]')
        return recipe

    def as_json(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def lr_factor(schedule, step, total):
    """Multiplier of every group's base rate at 0-based logical step `step` of `total`."""
    kind = schedule.get('kind') if isinstance(schedule, dict) else None
    if kind == 'constant' and set(schedule) == {'kind'}:
        return 1.
    if kind != 'warmup-cosine' or set(schedule) - {'kind', 'warmup_ratio', 'warmup_steps', 'min_ratio'}:
        raise TuneMathError('schedule must be {"kind": "constant"} or warmup-cosine with warmup_ratio|warmup_steps and min_ratio')
    floor = schedule.get('min_ratio', 0.)
    if type(floor) not in (int, float) or not 0 <= floor <= 1:
        raise TuneMathError('schedule min_ratio must lie in [0, 1]')
    warmup = schedule.get('warmup_steps')
    if warmup is None:
        ratio = schedule.get('warmup_ratio', 0.)
        if type(ratio) not in (int, float) or not 0 <= ratio < 1:
            raise TuneMathError('schedule warmup_ratio must lie in [0, 1)')
        warmup = math.ceil(ratio * total)
    if type(warmup) is not int or warmup < 0:
        raise TuneMathError('schedule warmup_steps must be a nonnegative integer')
    if step < warmup:
        return (step + 1) / warmup
    # The last step of the budget runs at the floor.
    progress = (step - warmup) / max(1, total - warmup - 1)
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * min(progress, 1.)))


def effective_factors(spec, values, bases):
    """``{m: (B, A)}`` with ``Delta W_m = B A`` of one resolved adapter (scaling folded once), any parameterization."""
    factors = {}
    for name in spec['parameter_names']:
        if spec['kind'] == 'free-lora':
            factors[name] = (values['lora_B.'+name] * (spec['alpha'] / spec['rank']), values['lora_A.'+name])
        else:
            core = values[name].reshape(*values[name].shape[:-1], spec['rank'], spec['rank'])
            factors[name] = (bases['left.'+name] @ core, bases['right.'+name].transpose(-2, -1))
    return factors


def artifact_member(engine, artifact, node, level, scale, bases):
    """One stored adapter (any basis, including a learned bank basis) as a `stack.StackMember`, adding its
    `bases.CanonicalBasis` views to ``bases``: the exact export path of tuner nodes and learned-optimizer releases."""
    from .bases import CanonicalBasis, tensor_digest
    from .parameterization import FreeLoRADelta, SubspaceDelta
    from .stack import StackMember
    record = engine.store.get_native_tensor_artifact(artifact).artifact
    spec = record.provenance['adapter_spec']
    deltas = {}
    for name in spec['parameter_names']:
        if spec['kind'] == 'free-lora':
            deltas[name] = FreeLoRADelta(record.tensors['lora_B.'+name], record.tensors['lora_A.'+name], spec['alpha'])
            continue
        key = f'{record.source_artifact_id}:{name}'
        if key not in bases:
            basis = engine.store.get_native_tensor_artifact(record.source_artifact_id).artifact
            left, right = basis.tensors['left.'+name], basis.tensors['right.'+name]
            bases[key] = CanonicalBasis(key, name, left.shape[-1], left, right, tensor_digest(left, right))
        rank = spec['rank']
        deltas[name] = SubspaceDelta(key, rank, record.tensors[name].reshape(*record.tensors[name].shape[:-1], rank, rank))
    return StackMember(node, level, scale, deltas)


def adapter_factors(engine, artifact_id):
    """The exact effective update factors of a stored native adapter (`effective_factors`)."""
    _, _, spec, values, bases = resolve_adapters(engine, [artifact_id])[0]
    return effective_factors(spec, values, bases)


def _tensor_hex(tensor):
    return [float(value).hex() for value in tensor.detach().double().reshape(-1).tolist()]


class JointTrainer:
    """Resumable joint trainer over one program manifest and its admitted training rows."""

    def __init__(self, engine, manifest, rows, recipe, *, state=None, owner=None, workdir=None, parent=None,
                 pinned=None, warm=None):
        """`rows` are SFT rows (`{binding, graph, ...}`) and RL rollouts (`natlang.tune-rollout/v1`); RL admission
        archives its behavior bundle and registered objective under `workdir`.

        `parent` is the final state of the previous fit of the same manifest (an earlier SFT or RL round). A new fit
        continues its node, soft-state and gate versions with their optimizer moments and lineage histories, so
        rollouts sampled from the parent's release are admitted as recorded-lineage behavior. Its own row set,
        schedule and step counter start fresh; cumulative exposure carries over.

        `pinned` maps scope nodes to existing coefficient artifacts that stay frozen (plan §8 incremental candidates and
        shared refits): they run in every stack that includes them, receive no gradient and are never updated, so the
        release carries the exact same artifacts. `warm` maps trainable nodes to `{artifact, optimizer?}` of an earlier
        compatible fit (same rank, parameterization, target names and canonical basis): the node starts from that
        artifact, and its optimizer moments continue only when the given state binds that exact artifact and the
        recipe's algorithm; otherwise its optimizer history starts new and the receipt says so."""
        self.engine, self.manifest, self.recipe, self.owner = engine, manifest, recipe, owner
        if manifest.get('schema') != 'natlang.adapters/v1':
            raise TuneMathError('trainer consumes a natlang.adapters/v1 manifest')
        if manifest['base'] != {key: engine.native_identity[key] for key in manifest['base']}:
            raise TuneMathError('manifest base differs from the exact live native model')
        self.nodes = {node['id']: node for node in manifest['tree']}
        self.bindings = {binding['id']: binding for binding in manifest['bindings']}
        self.families = {(item['family'], item['version']): item for item in manifest['families']}
        self.memberships = [item for item in manifest['memberships'] if item['binding'] in self.bindings]
        self.pinned, self.warm = dict(sorted((pinned or {}).items())), dict(sorted((warm or {}).items()))
        stack_nodes = set(self._stack_nodes())
        if set(self.pinned) - stack_nodes or set(self.warm) - stack_nodes:
            raise TuneMathError('pinned and warm-started nodes must be trained scope nodes of this manifest')
        if set(self.pinned) & set(self.warm):
            raise TuneMathError('a node is either pinned or warm-started, not both')
        self._soft_initial = {}
        from .rl import ROLLOUT_SCHEMA
        rollouts = [row for row in rows if isinstance(row, dict) and row.get('schema') == ROLLOUT_SCHEMA]
        sft = [row for row in rows if not (isinstance(row, dict) and row.get('schema') == ROLLOUT_SCHEMA)]
        episodic = {'rl', 'rft'} & set(recipe.objectives)
        if bool(rollouts) != bool(episodic) or bool(sft) != ('sft' in recipe.objectives):
            raise TuneMathError('the recipe objectives must match the admitted row kinds (sft rows; rollouts for rl/rft)')
        self.rows = self._admit(sft) if sft else []
        self.episodes = {}
        if rollouts:
            if workdir is None:
                raise TuneMathError('RL admission needs a work directory for its behavior bundle and objective')
            from .rl import RLEpisodes
            for kind in sorted(episodic):
                # Each objective archives its own behavior bundle and registered objective.
                self.episodes[kind] = RLEpisodes(engine, rollouts, Path(workdir) / kind if kind == 'rft' else workdir,
                                                 mode='grpo' if kind == 'rl' else 'rft', min_quality=recipe.rft_min_quality)
            self._admit_rl()
        available = target_parameters(engine.shared_model)
        self.targets, self.target_classes = module_map(sorted(available), classes=recipe.target_classes)
        self.target_names = sorted(self.targets)
        if not self.target_names:
            raise TuneMathError('the recipe selects no native target matrices')
        self.q = {name: 1. / available[name].numel() for name in self.target_names}
        self.state = state or (self._continued_state(parent) if parent is not None else self._initial_state())
        if self.state.get('schema') != STATE_SCHEMA or self.state['input_sha256'] != self.input_sha256:
            raise TuneMathError('trainer state belongs to another manifest, row set or recipe')
        # States written before lineage/RL cursors existed resume with their current versions as history.
        self.state.setdefault('node_history', {node: [artifact] for node, artifact in self.state['node_artifacts'].items()})
        self.state.setdefault('soft_history', {key: [ident] for key, ident in self.state.get('soft_states', {}).items()})
        for kind in ('rl', 'rft'):
            self.state.setdefault(f'{kind}_epoch', 0)
            self.state.setdefault(f'{kind}_position', 0)
        if self.episodes:
            self._check_behavior()
        self.gates = {key: torch.tensor(float.fromhex(value), dtype=torch.float32, requires_grad=True)
                      for key, value in self.state['gates'].items()}
        self.gate_optimizer = torch.optim.Adam(list(self.gates.values()), lr=recipe.gate_learning_rate) if self.gates else None
        if self.state.get('gate_optimizer') and self.gate_optimizer is not None:
            saved = self.state['gate_optimizer']
            self.gate_optimizer.load_state_dict({'state': {int(index): {'step': torch.tensor(float.fromhex(item['step'])),
                'exp_avg': torch.tensor(float.fromhex(item['exp_avg'])), 'exp_avg_sq': torch.tensor(float.fromhex(item['exp_avg_sq']))}
                for index, item in saved['state'].items()}, 'param_groups': saved['param_groups']})
        self._bases = {}

    # -- inputs --------------------------------------------------------------------------------------------------
    def _admit(self, rows):
        admitted, occurrences = [], set()
        for index, row in enumerate(rows):
            binding = row.get('binding')
            if binding not in self.bindings:
                raise TuneMathError(f'row {index} names unknown binding {binding}')
            graph = row.get('graph')
            if not isinstance(graph, dict) or graph.get('split') != 'train':
                raise TuneMathError(f'row {index} must carry a train-split native graph')
            weight = row.get('evidence_weight', 1.)
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
                raise TuneMathError(f'row {index} evidence weight must be finite and nonnegative')
            occurrence = row.get('occurrence', graph.get('id'))
            if occurrence in occurrences:
                raise TuneMathError(f'occurrence {occurrence} appears twice; each recorded occurrence is one loss row')
            occurrences.add(occurrence)
            example = parse_graph(graph, tokenizer_fingerprint=self.engine.model_config['tokenizer_fingerprint'])
            soft = row.get('soft') or {}
            node_bindings = row.get('bindings') or {}
            names = {node.name for node in example.plan.invocations}
            if (not isinstance(node_bindings, dict) or set(node_bindings) - names
                    or any(value is not None and value not in self.bindings for value in node_bindings.values())):
                raise TuneMathError(f'row {index} binds invocations it does not hold or unknown bindings')
            if example.external_states:
                # Only soft parameters of the row's own binding are imported here; any other stored state would be
                # an external input this trainer does not own.
                if set(soft) != set(example.external_states):
                    raise TuneMathError(f'row {index} imports stored state that is not one of its soft states; Neuralese '
                                        'between invocations belongs inside the row\'s graph, as producer and consumer nodes')
                for name, key in soft.items():
                    owner_binding, kind, target = json.loads(key)
                    if owner_binding != binding or kind not in ('instructions', 'skill'):
                        raise TuneMathError(f'row {index} imports a soft state of another binding')
                    initial = example.external_states[name]['id']
                    if self._soft_initial.setdefault(key, initial) != initial:
                        raise TuneMathError(f'soft state {key} has two initial states across rows')
            elif soft:
                raise TuneMathError(f'row {index} declares soft states it does not import')
            admitted.append({'binding': binding, 'graph': graph, 'example': example, 'evidence_weight': float(weight),
                             'occurrence': occurrence, 'soft': soft, **({'bindings': node_bindings} if node_bindings else {})})
        if not admitted:
            raise TuneMathError('no admitted training rows')
        return admitted

    @property
    def input_sha256(self):
        # The step budget is where to stop, not what a step does: extending it continues the same run.
        recipe = {key: value for key, value in self.recipe.as_json().items() if key not in ('steps', 'epochs')}
        return canonical_json_sha256_hex({'manifest': self.manifest, 'recipe': recipe,
            'rows': [[row['binding'], row['occurrence'], row['evidence_weight'], canonical_json_sha256_hex(row['graph']),
                      *([row['bindings']] if 'bindings' in row else [])] for row in self.rows], 'targets': self.target_names,
            **({'pinned': self.pinned} if self.pinned else {}), **({'warm': self.warm} if self.warm else {}),
            **{kind: episodes.digest() for kind, episodes in self.episodes.items()}})

    @property
    def total_steps(self):
        """The step budget: `steps`, or `epochs` passes over every objective's admitted rows. Without any objective
        that has admitted work, the fit makes no step (its receipt says why)."""
        if not self.cycle:
            return 0
        if self.recipe.epochs is None:
            return self.recipe.steps
        rows = len(self.rows) + sum(len(episodes.graphs) for episodes in self.episodes.values())
        return math.ceil(self.recipe.epochs * rows / min(self.recipe.batch_rows, max(rows, 1)))

    def _stack_nodes(self):
        """Every scope node with a residual in this release: trained here or pinned."""
        used = {node for binding in self.bindings.values() for node in binding['ancestry']}
        return sorted(node for node in used if self.recipe.ranks.get(self.nodes[node]['level'], 0) > 0)

    def _trained_nodes(self):
        return [node for node in self._stack_nodes() if node not in self.pinned]

    def _compatible(self, artifact_id, rank, basis_id):
        """Why an earlier artifact cannot seed this node, or None when it can."""
        record = self.engine.store.get_native_tensor_artifact(artifact_id).artifact
        spec = (record.provenance or {}).get('adapter_spec') or {}
        kind = 'free-lora' if self.recipe.parameterization == 'free-lora' else 'xs'
        if record.kind != 'adapter-coefficients' or spec.get('kind') != kind:
            return 'another parameterization'
        if spec.get('rank') != rank:
            return f"rank {spec.get('rank')} differs from {rank}"
        if list(spec.get('parameter_names', [])) != self.target_names:
            return 'other target matrices'
        if basis_id is not None and record.source_artifact_id != basis_id:
            return 'another canonical basis'
        return None

    def _initial_state(self):
        recipe, engine = self.recipe, self.engine
        nodes = self._trained_nodes()
        artifacts, basis_id = {}, None
        if recipe.parameterization == 'native-subspace':
            # The basis is a pure function of the base, maximum rank and targets: over every stack node, pinned ones
            # included, it is the same basis the earlier fit sliced, so pinned and warm artifacts compose exactly.
            max_rank = max(recipe.ranks[self.nodes[node]['level']] for node in self._stack_nodes())
            if recipe.basis is not None:
                record = engine.store.get_native_tensor_artifact(recipe.basis)
                spec = (record.artifact.provenance or {}).get('adapter_spec') if record else None
                if spec is None or spec.get('kind') != 'bank-basis':
                    raise TuneMathError(f'declared basis {recipe.basis} is not an available bank basis; import the bank first')
                if max_rank > spec['max_rank'] or not set(self.target_names) <= set(spec['parameter_names']):
                    raise TuneMathError('the fit\'s ranks and target matrices must fit inside the declared bank basis')
                basis_id = record.id
            else:
                basis_id = create_canonical_basis(engine, {'max_rank': max_rank, 'parameter_names': self.target_names}, self.owner).id
            for index, node in enumerate(nodes):
                # Zero cores of equal rank would otherwise be one content-addressed artifact shared by sibling nodes,
                # and one graph whose calls run both siblings could not tell their coefficients apart.
                artifacts[node] = create_adapter(engine, {'kind': 'xs', 'rank': recipe.ranks[self.nodes[node]['level']],
                    'seed': recipe.seed + index, 'parameter_names': self.target_names, 'basis_id': basis_id}, self.owner).id
        else:
            for index, node in enumerate(nodes):
                rank = recipe.ranks[self.nodes[node]['level']]
                artifacts[node] = create_adapter(engine, {'kind': 'free-lora', 'rank': rank, 'seed': recipe.seed + index,
                    'alpha': float(recipe.free_lora_alpha or rank), 'parameter_names': self.target_names}, self.owner).id
        for node, artifact in self.pinned.items():
            reason = self._compatible(artifact, recipe.ranks[self.nodes[node]['level']], basis_id)
            if reason is not None:
                raise TuneMathError(f'pinned node {node} cannot be composed with this fit: {reason}')
        optimizers, incremental = {node: None for node in nodes}, {'pinned': dict(self.pinned), 'warm': {}, 'cold': {}}
        for node in nodes:
            if node not in self.warm:
                continue
            seed = self.warm[node]
            reason = self._compatible(seed['artifact'], recipe.ranks[self.nodes[node]['level']], basis_id)
            if reason is not None:
                incremental['cold'][node] = reason
                continue
            artifacts[node] = seed['artifact']
            optimizer, history = seed.get('optimizer'), 'new'
            if optimizer is not None:
                old = engine.store.get_native_tensor_artifact(optimizer).artifact
                if old.source_artifact_id == seed['artifact'] and (old.provenance or {}).get('optimizer') == recipe.optimizer.get('name'):
                    optimizers[node], history = optimizer, 'continued'
            incremental['warm'][node] = {'artifact': seed['artifact'], 'optimizer': history}
        artifacts.update(self.pinned)
        gates = {}
        for item in self.memberships:
            if (item['family'], item['version']) not in self.families:
                raise TuneMathError(f'membership names unpublished family {item["family"]}@{item["version"]}')
            gates[json.dumps([item['binding'], item['family'], item['version']])] = float(min(max(item['gate'], 0.), self.recipe.a_max)).hex()
        if len(set(artifacts.values())) != len(artifacts):
            raise TuneMathError('every trained node needs its own coefficient artifact')
        state = {'schema': STATE_SCHEMA, 'step': 0, 'epoch': 0, 'position': 0, 'node_artifacts': dict(sorted(artifacts.items())),
                 'optimizer_states': optimizers, 'gates': gates, 'gate_optimizer': None,
                 'soft_states': dict(sorted(self._soft_initial.items())),
                 'soft_optimizer_states': {key: None for key in sorted(self._soft_initial)},
                 'node_history': {node: [artifact] for node, artifact in artifacts.items()},
                 'soft_history': {key: [ident] for key, ident in sorted(self._soft_initial.items())},
                 'rl_epoch': 0, 'rl_position': 0, 'exposure': {}, 'losses': []}
        if self.pinned or self.warm:
            state['incremental'] = incremental
        state['input_sha256'] = self.input_sha256
        return state

    def _continued_state(self, parent):
        if parent.get('schema') != STATE_SCHEMA:
            raise TuneMathError('a parent fit must be a tune trainer state')
        nodes = self._stack_nodes()
        if any(parent['node_artifacts'].get(node) != artifact for node, artifact in self.pinned.items()):
            raise TuneMathError('a continued fit pins a node at another artifact than its parent holds')
        if sorted(parent['node_artifacts']) != nodes or set(parent['gates']) != {
                json.dumps([item['binding'], item['family'], item['version']]) for item in self.memberships}:
            raise TuneMathError('the parent fit trained another node or gate set; continue only the same manifest and ranks')
        # Soft states first admitted in this fit start from their recorded initial version.
        soft = {**self._soft_initial, **parent.get('soft_states', {})}
        state = {'schema': STATE_SCHEMA, 'step': 0, 'epoch': 0, 'position': 0,
                 'node_artifacts': dict(parent['node_artifacts']),
                 'optimizer_states': {node: value for node, value in parent['optimizer_states'].items() if node not in self.pinned},
                 'gates': dict(parent['gates']), 'gate_optimizer': parent.get('gate_optimizer'),
                 'soft_states': dict(sorted(soft.items())),
                 'soft_optimizer_states': {key: parent.get('soft_optimizer_states', {}).get(key) for key in sorted(soft)},
                 'node_history': {node: list(parent.get('node_history', {}).get(node, [artifact]))
                                  for node, artifact in parent['node_artifacts'].items()},
                 'soft_history': {key: list(parent.get('soft_history', {}).get(key, [ident])) for key, ident in soft.items()},
                 'rl_epoch': 0, 'rl_position': 0, 'exposure': dict(parent.get('exposure', {})), 'losses': [],
                 'parent': {'input_sha256': parent['input_sha256'], 'step': parent['step']}}
        state['input_sha256'] = self.input_sha256
        return state

    # -- one logical update --------------------------------------------------------------------------------------
    def _order(self, epoch, count=None, salt=0):
        generator = torch.Generator().manual_seed(self.recipe.seed * 1000003 + epoch + salt * 7919)
        return torch.randperm(len(self.rows) if count is None else count, generator=generator).tolist()

    def _next_batch(self):
        batch, epoch, position = [], self.state['epoch'], self.state['position']
        order = self._order(epoch)
        while len(batch) < min(self.recipe.batch_rows, len(self.rows)):
            if position == len(order):
                epoch, position = epoch + 1, 0
                order = self._order(epoch)
            batch.append(order[position])
            position += 1
        return batch, epoch, position

    def _stack(self, binding_id):
        """Families (gate-scaled frozen tensors), then the ancestry root to leaf: the runtime's stack order."""
        binding = self.bindings[binding_id]
        families = sorted((item for item in self.memberships if item['binding'] == binding_id), key=lambda item: item['family'])
        scales = self.manifest.get('scales', {})
        return ([('family', item) for item in families] +
                [('node', node) for node in binding['ancestry']
                 if node in self.state['node_artifacts'] and scales.get(node, 1) != 0])

    def _basis(self, artifact_id):
        if artifact_id not in self._bases:
            self._bases[artifact_id] = resolve_adapters(self.engine, [artifact_id])[0]
        return self._bases[artifact_id]

    def _node_factors(self, artifact_id, values):
        _, _, spec, _, bases = self._basis(artifact_id)
        return effective_factors(spec, values, bases)

    # -- RL admission -------------------------------------------------------------------------------------------
    @property
    def rl(self):
        """The group-relative policy episodes (None without an "rl" objective)."""
        return self.episodes.get('rl')

    @property
    def _rollouts(self):
        return next(iter(self.episodes.values())).rollouts

    @property
    def cycle(self):
        """The recipe's objectives that have admitted work: an RL set without reward spread or an RFT set without a
        successful episode contributes no step (reported in the receipt)."""
        return [kind for kind in self.recipe.objectives
                if kind == 'sft' or (kind in self.episodes and self.episodes[kind].graphs)]

    def _admit_rl(self):
        for record in self._rollouts:
            unknown = set(record['bindings'].values()) - set(self.bindings) - {None}
            if unknown:
                raise TuneMathError(f'RL rollout names unknown bindings {sorted(unknown)}')
            for key, ident in record['behavior'].get('soft_states', {}).items():
                owner_binding, kind, _ = json.loads(key)
                if owner_binding not in record['bindings'].values() or kind not in ('instructions', 'skill'):
                    raise TuneMathError(f'RL rollout consumed soft state {key} of a binding it did not run')
                # A fresh fit starts from the first recorded version; later versions must descend from it.
                self._soft_initial.setdefault(key, ident)

    def _behavior_entries(self, record, binding_id, *, current):
        """The ordered stack of one binding, at the rollout's behavior versions or at the current ones."""
        behavior = record['behavior']
        entries = []
        if binding_id is None:
            return entries  # an invocation without a tuned binding (e.g. a rebound definition) runs the base
        for kind, item in self._stack(binding_id):
            if kind == 'family':
                key = json.dumps([item['binding'], item['family'], item['version']])
                if key not in behavior.get('gates', {}):
                    raise TuneMathError(f'RL rollout omits the behavior gate of {key}')
                raw = behavior['gates'][key]
                gate = float.fromhex(raw) if isinstance(raw, str) else float(raw)
                if gate <= 0:
                    continue  # the runtime serves no family at a zero gate
                entries.append({'id': self.families[(item['family'], item['version'])]['artifact'], 'scale': gate,
                                'family_key': key})
            else:
                artifact = self.state['node_artifacts'][item] if current else behavior['node_artifacts'].get(item)
                if artifact is None:
                    raise TuneMathError(f'RL rollout omits the behavior version of node {item}')
                entries.append({'id': artifact, 'scale': float(self.manifest.get('scales', {}).get(item, 1)), 'node': item})
        return entries

    def _check_behavior(self):
        """Every invocation's recorded actor is exactly its binding's stack at recorded versions of this run."""
        for record in self._rollouts:
            behavior = record['behavior']
            for node, artifact in behavior['node_artifacts'].items():
                if artifact not in self.state['node_history'].get(node, []):
                    raise TuneMathError(f'RL behavior version {artifact} of node {node} is not in this run\'s lineage')
            for key, ident in behavior.get('soft_states', {}).items():
                if ident not in self.state['soft_history'].get(key, []):
                    raise TuneMathError(f'RL behavior soft state {ident} is not in this run\'s lineage of {key}')
            for turn in record['episode']['turns']:
                expected = self._behavior_entries(record, record['bindings'][turn['invocation_id']], current=False)
                binding = (turn.get('sampling') or {}).get('adapter_binding')
                recorded = [] if binding is None else [(entry['id'], entry['scale_hex']) for entry in binding['adapters']]
                if recorded != [(entry['id'], float(entry['scale']).hex()) for entry in expected]:
                    raise TuneMathError('an RL turn was sampled by another actor than its binding\'s recorded stack')

    def _next_rl_batch(self, kind='rl'):
        graphs = self.episodes[kind].graphs
        batch, epoch, position = [], self.state[f'{kind}_epoch'], self.state[f'{kind}_position']
        salt = 1 if kind == 'rl' else 2
        order = self._order(epoch, len(graphs), salt=salt)
        while len(batch) < min(self.recipe.batch_rows, len(graphs)):
            if position == len(order):
                epoch, position = epoch + 1, 0
                order = self._order(epoch, len(graphs), salt=salt)
            batch.append(order[position])
            position += 1
        return batch, epoch, position

    def _graph_request(self, graph, binding_of, leaves, state_of, *, objective=None, behavior_gates=None):
        """The native gradient request of one whole graph -- an SFT row, an RL episode or an RFT episode alike: each
        invocation runs its binding's current stack (`binding_of(node name)`; None runs the base), nodes and family
        gates are live leaves, and every imported stored state is the current leaf of the soft state it is
        (`state_of(state id)`). One graph forward, gradients through every producer/consumer dependency.

        ``behavior_gates`` (RL/RFT) declares each family at its sampled gate: the leaf carries current/behavior, so the
        gate trains while the request matches the recorded actor. Without them a family is declared at scale 1 and
        its leaf carries the current gate."""
        engine = self.engine
        declarations, override, nodes = {}, {}, set()
        for node in graph['nodes']:
            entries = []
            binding = binding_of(node['name'])
            for kind, item in ([] if binding is None else self._stack(binding)):
                if kind == 'family':
                    key = json.dumps([item['binding'], item['family'], item['version']])
                    artifact = self.families[(item['family'], item['version'])]['artifact']
                    scale = 1.
                    if behavior_gates is not None:
                        if key not in behavior_gates:
                            raise TuneMathError(f'RL rollout omits the behavior gate of {key}')
                        raw = behavior_gates[key]
                        scale = float.fromhex(raw) if isinstance(raw, str) else float(raw)
                        if scale <= 0:
                            continue  # the runtime served no family at a zero gate
                    frozen = engine.store.get_native_tensor_artifact(artifact).artifact.tensors
                    ratio = self.gates[key] / scale
                    # A gate multiplies the whole frozen delta: the core, or the B factor of free LoRA.
                    override[artifact] = {name: value * ratio if not name.startswith('lora_A.') else value
                                          for name, value in frozen.items()}
                    entries.append({'id': artifact, 'scale': scale})
                else:
                    artifact = self.state['node_artifacts'][item]
                    override[artifact] = leaves[item]
                    nodes.add(item)
                    entries.append({'id': artifact, 'scale': float(self.manifest.get('scales', {}).get(item, 1))})
            declarations[node['name']] = entries
        imported = sorted({binding['id'] for binding in graph.get('external_states', {}).values()})
        states = {ident: state_of(ident) for ident in imported}
        node_ids = {self.state['node_artifacts'][node] for node in nodes}
        specs = {f'node:{node}': {'kind': 'adapter', 'id': self.state['node_artifacts'][node]} for node in sorted(nodes)}
        specs.update({f'family:{ident}': {'kind': 'adapter', 'id': ident, 'tensors': tensors}
                      for ident, tensors in override.items() if ident not in node_ids})
        specs.update({f'soft:{ident}': {'kind': 'native-state', 'id': ident} for ident in states})
        body = {'schema': 'natlang.native-gradient-request/1', 'native_identity': engine.native_identity, 'graph': graph,
                'invocation_adapters': declarations, 'adapter_gradient_precision': 'fp32',
                **({'objective': objective} if objective is not None else {}),
                **({'native_state_modes': {ident: 'same-base-adapter-state' for ident in states}} if states else {})}
        return body, {'adapter_leaves_override': override, 'native_state_overrides': states or None,
                      'evaluation_target_specs': specs}

    def _episode_request(self, graph, leaves, soft_blocks, kind='rl'):
        """An admitted episode's whole graph with the current node versions, gates and soft states substituted for its
        behavior versions on the same bases."""
        record = self.episodes[kind].rollout(graph)
        behavior_soft = record['behavior'].get('soft_states', {})
        owned = {ident: key for key, ident in behavior_soft.items()}
        def state_of(ident):
            if ident not in owned:
                raise TuneMathError('an RL episode imports stored state this run does not own')
            return soft_blocks[owned[ident]]
        return self._graph_request(graph, lambda name: record['bindings'][call_of(name)], leaves, state_of,
                                   objective=self.episodes[kind].objective, behavior_gates=record['behavior'].get('gates', {}))

    def _row_request(self, row, leaves, soft_blocks):
        """A supervised row's graph (one turn, or several invocations with their own bindings) through the same request."""
        imported = {binding['id']: row['soft'][name] for name, binding in row['graph'].get('external_states', {}).items()}
        bindings = row.get('bindings') or {}
        return self._graph_request(row['graph'], lambda name: bindings.get(name, row['binding']), leaves,
                                   lambda ident: soft_blocks[imported[ident]])

    def _backward(self, requests, weights):
        """Accumulate ``sum_i weight_i * loss_i`` over whole-graph requests into the snapshot leaves; returns the losses.
        A graph whose whole stack is pinned has no gradient path and trains nothing (incremental fits)."""
        from ..serve.native_grad import native_graph_gradient
        losses = []
        for (body, overrides), weight in zip(requests, weights):
            def evaluate(loss, groups, meta, weight=weight):
                if loss.requires_grad:
                    (loss * weight).backward()
                return float(loss.detach())
            losses.append(native_graph_gradient(self.engine, body, self.owner, evaluation_callback=evaluate, **overrides))
        return losses

    def _rl_backward(self, batch, leaves, soft_leaves, kind='rl'):
        soft = {key: value[1] for key, value in soft_leaves.items()}
        return self._backward([self._episode_request(self.episodes[kind].graphs[index], leaves, soft, kind) for index in batch],
                              [1. / len(batch)] * len(batch))

    def rl_losses(self, kind='rl'):
        """Current per-episode losses of one episodic objective without an update, for diagnostics and tests."""
        from ..serve.native_grad import _leaf_block, native_graph_gradient
        store = self.engine.store
        leaves = {node: store.get_native_tensor_artifact(artifact).artifact.tensors
                  for node, artifact in self.state['node_artifacts'].items()}
        soft = {key: _leaf_block(store.get_native_state(ident).block)[0] for key, ident in self.state['soft_states'].items()}
        losses = {}
        for graph in self.episodes[kind].graphs:
            body, overrides = self._episode_request(graph, leaves, soft, kind)
            losses[graph['id']] = native_graph_gradient(self.engine, body, self.owner,
                evaluation_callback=lambda loss, groups, meta: float(loss.detach()), **overrides)
        return losses

    # -- one logical update --------------------------------------------------------------------------------------
    def step(self):
        engine, recipe = self.engine, self.recipe
        cycle = self.cycle
        if not cycle:
            raise TuneMathError('no objective of the recipe has admitted work (no rows, no reward spread, no success)')
        objective = cycle[self.state['step'] % len(cycle)]
        factor = lr_factor(recipe.schedule, self.state['step'], max(self.total_steps, 1))
        # One immutable snapshot of every node for the whole batch.
        leaves = {node: {key: value.detach().clone().requires_grad_(node not in self.pinned)
                         for key, value in engine.store.get_native_tensor_artifact(artifact).artifact.tensors.items()}
                  for node, artifact in self.state['node_artifacts'].items()}
        from ..serve.native_grad import _leaf_block
        if objective == 'sft':
            batch, epoch, position = self._next_batch()
            active_soft = sorted({key for index in batch for key in self.rows[index]['soft'].values()})
            bindings_used = [self.rows[index]['binding'] for index in batch]
        else:
            batch, epoch, position = self._next_rl_batch(objective)
            episodes = self.episodes[objective]
            records = [episodes.rollout(episodes.graphs[index]) for index in batch]
            active_soft = sorted({key for record in records for key, ident in record['behavior'].get('soft_states', {}).items()})
            # Unbound invocations execute the frozen base. They stay in the
            # complete policy graph; only adapter-parameter routing omits them.
            bindings_used = sorted({binding for record in records for binding in record['bindings'].values()
                                    if binding is not None})
        soft_leaves = {}
        for key in active_soft:
            block = engine.store.get_native_state(self.state['soft_states'][key]).block
            copied, values = _leaf_block(block)
            soft_leaves[key] = (block, copied, values)
        per_binding = {}
        for binding in bindings_used:
            per_binding[binding] = per_binding.get(binding, 0) + 1
        row_losses = []
        if objective != 'sft':
            row_losses = [float(result) for result in self._rl_backward(batch, leaves, soft_leaves, objective)]
        if objective == 'sft':
            # §5.1: mean over lambdas of the mean evidence-weighted row loss, as an exact per-row coefficient.
            soft = {key: value[1] for key, value in soft_leaves.items()}
            rows = [self.rows[index] for index in batch]
            row_losses = [float(loss) for loss in self._backward(
                [self._row_request(row, leaves, soft) for row in rows],
                [row['evidence_weight'] / (len(per_binding) * per_binding[row['binding']]) for row in rows])]
        # Pinned nodes run in the stacks but are neither penalized nor updated.
        # Every binding that ran in a batch graph trains, whether its loss is its own or reached it through an edge.
        ran = set(per_binding) | ({binding for index in batch for binding in (self.rows[index].get('bindings') or {}).values()
                                   if binding is not None} if objective == 'sft' else set())
        active_nodes = sorted({node for binding in ran for kind, node in self._stack(binding)
                               if kind == 'node' and node not in self.pinned})
        penalty = torch.zeros(())
        for node in active_nodes:
            level = self.nodes[node]['level']
            factors = self._node_factors(self.state['node_artifacts'][node], leaves[node])
            penalty = penalty + recipe.depth_penalty[level] * sum(self.q[name] * factor_sq_norm(B, A) for name, (B, A) in factors.items())
        active_gates = {key: value for key, value in self.gates.items() if json.loads(key)[0] in ran}
        by_binding = {}
        for key, value in active_gates.items():
            by_binding.setdefault(json.loads(key)[0], []).append(value)
        gate_term = gate_l1({binding: torch.stack(values) for binding, values in by_binding.items()}, recipe.tau)
        if penalty.requires_grad or gate_term.requires_grad:
            (penalty + gate_term).backward()
        # One optimizer step per active node, through the immutable lineage/continuation contract, at its group's
        # scheduled rate.
        updates, rates, restarted = {}, {}, []
        for node in active_nodes:
            grads = {key: value.grad for key, value in leaves[node].items() if value.grad is not None}
            if not grads:
                continue
            artifact = self.state['node_artifacts'][node]
            gradient = NativeTensorArtifact('adapter-gradient', engine.native_identity,
                                            {key: grad.detach().cpu() for key, grad in grads.items()}, artifact)
            options = dict(recipe.optimizer)
            options['learning_rate'] = rates[node] = (options.get('learning_rate', 1e-3) * factor
                                                      * recipe.level_lr_scale[self.nodes[node]['level']])
            if options['learning_rate'] <= 0:
                continue
            previous = self.state['optimizer_states'][node]
            if previous is not None and (engine.store.get_native_tensor_artifact(previous).artifact.provenance or {}).get(
                    'optimizer') != options.get('name'):
                # A continued fit that changes the algorithm starts honestly new moments; the old state stays recorded.
                restarted.append(node)
                previous = None
            result = optimize_adapter(engine, gradient, options, previous, self.owner)
            updates[node] = (result['adapter']['id'], result['optimizer_state_id'])
        soft_updates = {}
        from ..model.native_state_derivation import optimize_state
        for key in active_soft:
            grads = {name: value.grad for name, value in soft_leaves[key][2].items() if value.grad is not None}
            if not grads:
                continue
            parent = self.state['soft_states'][key]
            gradient = engine.store.put_native_tensor_artifact(NativeTensorArtifact('state-gradient', engine.native_identity,
                {name: grad.detach().cpu() for name, grad in grads.items()}, parent,
                {'scope': 'natlang.tune.soft-state/1', 'step': self.state['step'] + 1, 'soft_key': key}), self.owner)
            options = dict(recipe.state_optimizer)
            options['learning_rate'] = rates['soft:' + key] = options.get('learning_rate', 1e-3) * factor
            result, _, _ = optimize_state(engine, gradient, options, self.state['soft_optimizer_states'][key], self.owner)
            soft_updates[key] = (result['native_state_id'], result['optimizer_state_id'])
        if any(value.grad is not None for value in self.gates.values()):
            for group in self.gate_optimizer.param_groups:
                group['lr'] = recipe.gate_learning_rate * factor
            self.gate_optimizer.step()
            project_gates(self.gates, recipe.a_max)
            self.gate_optimizer.zero_grad(set_to_none=True)
        for node, (artifact, optimizer_state) in updates.items():
            self.state['node_artifacts'][node], self.state['optimizer_states'][node] = artifact, optimizer_state
            self.state['node_history'].setdefault(node, []).append(artifact)
        for key, (state_id, optimizer_state) in soft_updates.items():
            self.state['soft_states'][key], self.state['soft_optimizer_states'][key] = state_id, optimizer_state
            self.state['soft_history'].setdefault(key, []).append(state_id)
        if objective == 'sft':
            for index in batch:
                occurrence = self.rows[index]['occurrence']
                self.state['exposure'][occurrence] = self.state['exposure'].get(occurrence, 0) + 1
            self.state.update(epoch=epoch, position=position)
        else:
            for index in batch:
                occurrence = f'{objective}:' + self.episodes[objective].graphs[index]['id']
                self.state['exposure'][occurrence] = self.state['exposure'].get(occurrence, 0) + 1
            self.state.update({f'{objective}_epoch': epoch, f'{objective}_position': position})
        self.state['step'] += 1
        self.state['gates'] = {key: float(value.detach()).hex() for key, value in self.gates.items()}
        if self.gate_optimizer is not None:
            saved = self.gate_optimizer.state_dict()
            self.state['gate_optimizer'] = {'param_groups': saved['param_groups'], 'state': {str(index): {
                key: float(value).hex() for key, value in item.items()} for index, item in saved['state'].items()}}
        record = {'step': self.state['step'], 'objective': objective, 'lr_factor': factor, 'learning_rates': rates,
                  'row_losses': row_losses, 'penalty': float(penalty.detach()),
                  'gate_l1': float(gate_term.detach()), 'active_nodes': active_nodes, 'updated_nodes': sorted(updates),
                  'updated_soft_states': sorted(soft_updates),
                  **({'optimizer_restarted': restarted} if restarted else {})}
        self.state['losses'].append(record)
        return record

    def run(self, steps=None):
        return [self.step() for _ in range(self.total_steps - self.state['step'] if steps is None else steps)]

    # -- outputs -------------------------------------------------------------------------------------------------
    def stack_members(self, binding_id):
        """The binding's effective stack as `stack.StackMember`s over `bases.CanonicalBasis` views of the native
        basis artifacts, for exact export (`stack.stack_factors`, `export.export_peft`)."""
        return self._members(self._stack(binding_id))

    def node_members(self, node):
        """One scope node's residual alone (scale from the manifest), for per-node backend exports that the server
        sums per request."""
        import dataclasses
        members, bases = self._members([('node', node)])
        # The serving request applies the manifest scale; the export carries the unscaled residual.
        return [dataclasses.replace(member, scale=1.) for member in members], bases

    def _members(self, items):
        members, bases = [], {}
        for kind, item in items:
            if kind == 'family':
                family = self.families[(item['family'], item['version'])]
                artifact, node, level = family['artifact'], f'family:{item["family"]}@{item["version"]}', 'family'
                scale = float.fromhex(self.state['gates'][json.dumps([item['binding'], item['family'], item['version']])])
            else:
                artifact, node, level = self.state['node_artifacts'][item], item, self.nodes[item]['level']
                scale = float(self.manifest.get('scales', {}).get(item, 1))
            members.append(artifact_member(self.engine, artifact, node, level, scale, bases))
        return members, bases

    def manifest_fragment(self):
        """The `artifacts` and `memberships` entries of a natlang.adapters/v1 release for the trained stack."""
        artifacts = []
        for node, artifact in sorted(self.state['node_artifacts'].items()):
            ancestors, parent = [], self.nodes[node]['parent']
            while parent is not None:
                if parent in self.state['node_artifacts']:
                    ancestors.append(self.state['node_artifacts'][parent])
                parent = self.nodes[parent]['parent']
            artifacts.append({'node': node, 'artifact': artifact, 'rank': self.recipe.ranks[self.nodes[node]['level']],
                              'parameterization': self.recipe.parameterization, 'requires': ancestors[::-1]})
        memberships = [{'binding': binding, 'family': family, 'version': version, 'gate': float.fromhex(value)}
                       for key, value in sorted(self.state['gates'].items()) for binding, family, version in [json.loads(key)]]
        return {'artifacts': artifacts, 'memberships': memberships}

    def soft_fragment(self, records):
        """The release's `softStates`: each initialized soft state's current tuned ID with its text digest."""
        result = []
        for key, state_id in sorted(self.state.get('soft_states', {}).items()):
            record = records.get(key)
            if record is None:
                raise TuneMathError(f'soft state {key} has no initialization record')
            block = self.engine.store.get_native_state(state_id).block
            if len(block.token_ids) != record['tokens']:
                raise TuneMathError(f'soft state {key} changed its token count')
            result.append({'binding': record['binding'], 'kind': record['kind'], 'target': record['target'],
                           'state': state_id, 'textSha256': record['textSha256'], 'tokens': record['tokens']})
        return result

    def receipt(self):
        exposure = self.state['exposure']
        sft = {key: count for key, count in exposure.items() if not key.startswith(('rl:', 'rft:'))}
        rl = {key: count for key, count in exposure.items() if key.startswith('rl:')}
        rft = {key: count for key, count in exposure.items() if key.startswith('rft:')}
        return {'schema': 'natlang.tune.training-receipt/1', 'input_sha256': self.input_sha256,
                'base_native_identity': dict(self.engine.native_identity), 'recipe': self.recipe.as_json(),
                'schedule': {**self.recipe.schedule, 'total_steps': self.total_steps,
                             'derived_from': 'epochs' if self.recipe.epochs is not None else 'steps'},
                'step': self.state['step'], 'epoch': self.state['epoch'], 'targets': self.target_names,
                'target_classes': self.target_classes, 'parent': self.state.get('parent'),
                'exposure': {'rows_seen': sum(sft.values()), 'distinct_occurrences': len(sft),
                             'by_binding': {binding: sum(count for occurrence, count in sft.items()
                                                         if any(row['occurrence'] == occurrence and row['binding'] == binding for row in self.rows))
                                            for binding in sorted({row['binding'] for row in self.rows})},
                             'rl_episodes_seen': sum(rl.values()), 'rl_distinct_episodes': len(rl),
                             **({'rft_episodes_seen': sum(rft.values()), 'rft_distinct_episodes': len(rft)} if 'rft' in self.episodes else {})},
                **({'rl': {'rollouts': len(self.rl.rollouts), 'admitted_episodes': len(self.rl.graphs),
                           'skipped_without_reward_spread': self.rl.skipped, 'objective': self.rl.objective}}
                   if self.rl is not None else {}),
                **({'rft': {'rollouts': len(self.episodes['rft'].rollouts), 'admitted_episodes': len(self.episodes['rft'].graphs),
                            'skipped_below_min_quality': self.episodes['rft'].skipped, 'min_quality': self.recipe.rft_min_quality,
                            'objective': self.episodes['rft'].objective}} if 'rft' in self.episodes else {}),
                'objectiveCycle': self.cycle,
                'losses': self.state['losses'], **self.manifest_fragment(),
                'soft_states': dict(self.state.get('soft_states', {})),
                'optimizer_states': self.state['optimizer_states'],
                **({'incremental': self.state['incremental']} if self.state.get('incremental') else {})}
