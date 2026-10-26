"""Joint path-routed program adapter training (plans/HIERARCHICAL_LORA_TUNING.md §5.1, §5.3, §5.4).

One logical update trains every tree/folder/lambda node of a program and its per-binding family gates together.
Each row runs the existing native graph objective (`data.native_graph.graph_loss`: one graph forward, positive-weight
normalized CE over its own complete trajectory) under its binding's own stack through `run_native`, with the current
node coefficients substituted as differentiable leaves. All rows of a batch accumulate gradients against one
immutable snapshot; only then does each active node take one optimizer step through `optimize_adapter`, which keeps
the immutable coefficient lineage and optimizer continuation contract. Published family tensors stay frozen: a gate
scales its family's frozen tensors, so the gate alone receives the gradient.

Native-subspace nodes slice one shared canonical maximum-rank basis per target matrix, so a stack sums exactly in that
basis. Free-LoRA nodes carry their own factors.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import torch

from ..common.hashing import canonical_json_sha256_hex
from ..data.native_graph import graph_loss
from ..data.native_graph_schema import parse_graph
from ..model.native_adapters import (create_adapter, create_canonical_basis, optimize_adapter, resolve_adapters,
                                     run_native, target_parameters)
from ..model.native_artifact import NativeTensorArtifact
from . import TuneMathError
from .gates import factor_sq_norm, gate_l1, project_gates
from .targets import module_map

STATE_SCHEMA = 'natlang.tune.trainer-state/1'
LEVELS = ('tree', 'folder', 'lambda')


@dataclass
class Recipe:
    steps: int = 1
    batch_rows: int = 8
    seed: int = 0
    parameterization: str = 'native-subspace'
    ranks: dict = field(default_factory=lambda: {'tree': 8, 'folder': 4, 'lambda': 2})
    depth_penalty: dict = field(default_factory=lambda: {'tree': 1e-4, 'folder': 1e-4, 'lambda': 1e-4})
    tau: float = 1e-3
    a_max: float = 1.0
    optimizer: dict = field(default_factory=lambda: {'name': 'adam', 'learning_rate': 1e-3})
    gate_learning_rate: float = 1e-2
    target_classes: list | None = None
    free_lora_alpha: float | None = None
    # Soft instruction/skill states (plan §3.3): the shared native-state optimizer's settings.
    state_optimizer: dict = field(default_factory=lambda: {'name': 'adam', 'learning_rate': 1e-3})

    @classmethod
    def from_json(cls, value):
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise TuneMathError(f'unknown recipe fields {sorted(unknown)}')
        recipe = cls(**value)
        if recipe.parameterization not in ('native-subspace', 'free-lora'):
            raise TuneMathError('parameterization must be native-subspace or free-lora')
        if set(recipe.ranks) - set(LEVELS) or any(type(rank) is not int or rank < 0 for rank in recipe.ranks.values()):
            raise TuneMathError('ranks map scope levels to nonnegative integers (0: no adapter at that level)')
        if any(not math.isfinite(recipe.depth_penalty.get(level, math.nan)) or recipe.depth_penalty[level] < 0
               for level, rank in recipe.ranks.items() if rank):
            raise TuneMathError('every trained level needs a finite nonnegative depth penalty')
        if type(recipe.steps) is not int or recipe.steps < 0 or type(recipe.batch_rows) is not int or recipe.batch_rows < 1:
            raise TuneMathError('steps and batch_rows must be positive integers')
        return recipe

    def as_json(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _tensor_hex(tensor):
    return [float(value).hex() for value in tensor.detach().double().reshape(-1).tolist()]


class JointTrainer:
    """Resumable joint trainer over one program manifest and its admitted training rows."""

    def __init__(self, engine, manifest, rows, recipe, *, state=None, owner=None):
        self.engine, self.manifest, self.recipe, self.owner = engine, manifest, recipe, owner
        if manifest.get('schema') != 'natlang.adapters/v1':
            raise TuneMathError('trainer consumes a natlang.adapters/v1 manifest')
        if manifest['base'] != {key: engine.native_identity[key] for key in manifest['base']}:
            raise TuneMathError('manifest base differs from the exact live native model')
        self.nodes = {node['id']: node for node in manifest['tree']}
        self.bindings = {binding['id']: binding for binding in manifest['bindings']}
        self.families = {(item['family'], item['version']): item for item in manifest['families']}
        self.memberships = [item for item in manifest['memberships'] if item['binding'] in self.bindings]
        self._soft_initial = {}
        self.rows = self._admit(rows)
        available = target_parameters(engine.shared_model)
        self.targets, self.target_classes = module_map(sorted(available), classes=recipe.target_classes)
        self.target_names = sorted(self.targets)
        if not self.target_names:
            raise TuneMathError('the recipe selects no native target matrices')
        self.q = {name: 1. / available[name].numel() for name in self.target_names}
        self.state = state or self._initial_state()
        if self.state.get('schema') != STATE_SCHEMA or self.state['input_sha256'] != self.input_sha256:
            raise TuneMathError('trainer state belongs to another manifest, row set or recipe')
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
            if example.external_states:
                # Only soft parameters of the row's own binding are imported here; any other stored state would be
                # an external input this trainer does not own.
                if set(soft) != set(example.external_states):
                    raise NotImplementedError('rows importing stored external native state other than their binding\'s '
                                              'soft states need per-occurrence state receipts; train them through native_grad')
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
                             'occurrence': occurrence, 'soft': soft})
        if not admitted:
            raise TuneMathError('no admitted training rows')
        return admitted

    @property
    def input_sha256(self):
        # The step budget is where to stop, not what a step does: extending it continues the same run.
        recipe = {key: value for key, value in self.recipe.as_json().items() if key != 'steps'}
        return canonical_json_sha256_hex({'manifest': self.manifest, 'recipe': recipe,
            'rows': [[row['binding'], row['occurrence'], row['evidence_weight'], canonical_json_sha256_hex(row['graph'])]
                     for row in self.rows], 'targets': self.target_names})

    def _trained_nodes(self):
        used = {node for binding in self.bindings.values() for node in binding['ancestry']}
        return sorted(node for node in used if self.recipe.ranks.get(self.nodes[node]['level'], 0) > 0)

    def _initial_state(self):
        recipe, engine = self.recipe, self.engine
        nodes = self._trained_nodes()
        artifacts = {}
        if recipe.parameterization == 'native-subspace':
            max_rank = max(recipe.ranks[self.nodes[node]['level']] for node in nodes)
            basis = create_canonical_basis(engine, {'max_rank': max_rank, 'parameter_names': self.target_names}, self.owner)
            for node in nodes:
                artifacts[node] = create_adapter(engine, {'kind': 'xs', 'rank': recipe.ranks[self.nodes[node]['level']],
                    'seed': recipe.seed, 'parameter_names': self.target_names, 'basis_id': basis.id}, self.owner).id
        else:
            for index, node in enumerate(nodes):
                rank = recipe.ranks[self.nodes[node]['level']]
                artifacts[node] = create_adapter(engine, {'kind': 'free-lora', 'rank': rank, 'seed': recipe.seed + index,
                    'alpha': float(recipe.free_lora_alpha or rank), 'parameter_names': self.target_names}, self.owner).id
        gates = {}
        for item in self.memberships:
            if (item['family'], item['version']) not in self.families:
                raise TuneMathError(f'membership names unpublished family {item["family"]}@{item["version"]}')
            gates[json.dumps([item['binding'], item['family'], item['version']])] = float(min(max(item['gate'], 0.), self.recipe.a_max)).hex()
        state = {'schema': STATE_SCHEMA, 'step': 0, 'epoch': 0, 'position': 0, 'node_artifacts': artifacts,
                 'optimizer_states': {node: None for node in nodes}, 'gates': gates, 'gate_optimizer': None,
                 'soft_states': dict(sorted(self._soft_initial.items())),
                 'soft_optimizer_states': {key: None for key in sorted(self._soft_initial)},
                 'exposure': {}, 'losses': []}
        state['input_sha256'] = self.input_sha256
        return state

    # -- one logical update --------------------------------------------------------------------------------------
    def _order(self, epoch):
        generator = torch.Generator().manual_seed(self.recipe.seed * 1000003 + epoch)
        return torch.randperm(len(self.rows), generator=generator).tolist()

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
        factors = {}
        for name in spec['parameter_names']:
            if spec['kind'] == 'free-lora':
                factors[name] = (values['lora_B.'+name] * (spec['alpha'] / spec['rank']), values['lora_A.'+name])
            else:
                core = values[name].reshape(*values[name].shape[:-1], spec['rank'], spec['rank'])
                factors[name] = (bases['left.'+name] @ core, bases['right.'+name].transpose(-2, -1))
        return factors

    def step(self):
        engine, recipe = self.engine, self.recipe
        batch, epoch, position = self._next_batch()
        # One immutable snapshot of every node for the whole batch.
        leaves = {node: {key: value.detach().clone().requires_grad_(True)
                         for key, value in engine.store.get_native_tensor_artifact(artifact).artifact.tensors.items()}
                  for node, artifact in self.state['node_artifacts'].items()}
        from ..serve.native_grad import _leaf_block
        from ..model.native_adapters import adapter_state_input_receipt
        active_soft = sorted({key for index in batch for key in self.rows[index]['soft'].values()})
        soft_leaves = {}
        for key in active_soft:
            block = engine.store.get_native_state(self.state['soft_states'][key]).block
            copied, values = _leaf_block(block)
            soft_leaves[key] = (block, copied, values)
        per_binding = {}
        for index in batch:
            per_binding[self.rows[index]['binding']] = per_binding.get(self.rows[index]['binding'], 0) + 1
        row_losses = []
        for index in batch:
            row = self.rows[index]
            entries, row_leaves = [], {}
            for kind, item in self._stack(row['binding']):
                if kind == 'family':
                    family = self.families[(item['family'], item['version'])]
                    gate = self.gates[json.dumps([item['binding'], item['family'], item['version']])]
                    frozen = engine.store.get_native_tensor_artifact(family['artifact']).artifact.tensors
                    # A gate multiplies the whole frozen delta: the core, or the B factor of free LoRA.
                    row_leaves[family['artifact']] = {key: value * gate if not key.startswith('lora_A.') else value
                                                      for key, value in frozen.items()}
                    entries.append({'id': family['artifact'], 'scale': 1.})
                else:
                    artifact = self.state['node_artifacts'][item]
                    row_leaves[artifact] = leaves[item]
                    entries.append({'id': artifact, 'scale': float(self.manifest.get('scales', {}).get(item, 1))})
            # §5.1: mean over lambdas of the mean evidence-weighted row loss, as an exact per-row coefficient.
            coefficient = row['evidence_weight'] / (len(per_binding) * per_binding[row['binding']])
            def execute(example=row['example'], soft=row['soft']):
                blocks = {name: soft_leaves[key][1] for name, key in soft.items()}
                identities = {}
                for name, key in soft.items():
                    source = soft_leaves[key][0]
                    # Same-base adapter-state exchange: the base-written soft state read by this binding's actor.
                    identities[name] = adapter_state_input_receipt(engine, source, self.state['soft_states'][key],
                                                                   target_position_start=0)['source_native_identity']
                loss, _ = graph_loss(engine.shared_model, example, external_blocks=blocks or None,
                                     external_identity=identities or None)
                (loss * coefficient).backward()
                return loss.detach()
            row_losses.append(float(run_native(engine, execute, entries, leaves=row_leaves, gradient_precision='fp32')))
        active_nodes = sorted({node for index in batch for kind, node in self._stack(self.rows[index]['binding']) if kind == 'node'})
        penalty = torch.zeros(())
        for node in active_nodes:
            level = self.nodes[node]['level']
            factors = self._node_factors(self.state['node_artifacts'][node], leaves[node])
            penalty = penalty + recipe.depth_penalty[level] * sum(self.q[name] * factor_sq_norm(B, A) for name, (B, A) in factors.items())
        active_gates = {key: value for key, value in self.gates.items() if json.loads(key)[0] in per_binding}
        by_binding = {}
        for key, value in active_gates.items():
            by_binding.setdefault(json.loads(key)[0], []).append(value)
        gate_term = gate_l1({binding: torch.stack(values) for binding, values in by_binding.items()}, recipe.tau)
        if penalty.requires_grad or gate_term.requires_grad:
            (penalty + gate_term).backward()
        # One optimizer step per active node, through the immutable lineage/continuation contract.
        updates = {}
        for node in active_nodes:
            grads = {key: value.grad for key, value in leaves[node].items() if value.grad is not None}
            if not grads:
                continue
            artifact = self.state['node_artifacts'][node]
            gradient = NativeTensorArtifact('adapter-gradient', engine.native_identity,
                                            {key: grad.detach().cpu() for key, grad in grads.items()}, artifact)
            result = optimize_adapter(engine, gradient, dict(recipe.optimizer), self.state['optimizer_states'][node], self.owner)
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
            result, _, _ = optimize_state(engine, gradient, dict(recipe.state_optimizer),
                                          self.state['soft_optimizer_states'][key], self.owner)
            soft_updates[key] = (result['native_state_id'], result['optimizer_state_id'])
        if any(value.grad is not None for value in self.gates.values()):
            self.gate_optimizer.step()
            project_gates(self.gates, recipe.a_max)
            self.gate_optimizer.zero_grad(set_to_none=True)
        for node, (artifact, optimizer_state) in updates.items():
            self.state['node_artifacts'][node], self.state['optimizer_states'][node] = artifact, optimizer_state
        for key, (state_id, optimizer_state) in soft_updates.items():
            self.state['soft_states'][key], self.state['soft_optimizer_states'][key] = state_id, optimizer_state
        for index in batch:
            occurrence = self.rows[index]['occurrence']
            self.state['exposure'][occurrence] = self.state['exposure'].get(occurrence, 0) + 1
        self.state.update(step=self.state['step'] + 1, epoch=epoch, position=position)
        self.state['gates'] = {key: float(value.detach()).hex() for key, value in self.gates.items()}
        if self.gate_optimizer is not None:
            saved = self.gate_optimizer.state_dict()
            self.state['gate_optimizer'] = {'param_groups': saved['param_groups'], 'state': {str(index): {
                key: float(value).hex() for key, value in item.items()} for index, item in saved['state'].items()}}
        record = {'step': self.state['step'], 'row_losses': row_losses, 'penalty': float(penalty.detach()),
                  'gate_l1': float(gate_term.detach()), 'active_nodes': active_nodes, 'updated_nodes': sorted(updates),
                  'updated_soft_states': sorted(soft_updates)}
        self.state['losses'].append(record)
        return record

    def run(self, steps=None):
        return [self.step() for _ in range(self.recipe.steps if steps is None else steps)]

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
        from .bases import CanonicalBasis, tensor_digest
        from .parameterization import FreeLoRADelta, SubspaceDelta
        from .stack import StackMember
        members, bases = [], {}
        for kind, item in items:
            if kind == 'family':
                family = self.families[(item['family'], item['version'])]
                artifact, node, level = family['artifact'], f'family:{item["family"]}@{item["version"]}', 'family'
                scale = float.fromhex(self.state['gates'][json.dumps([item['binding'], item['family'], item['version']])])
            else:
                artifact, node, level = self.state['node_artifacts'][item], item, self.nodes[item]['level']
                scale = float(self.manifest.get('scales', {}).get(item, 1))
            record = self.engine.store.get_native_tensor_artifact(artifact).artifact
            spec = record.provenance['adapter_spec']
            deltas = {}
            for name in spec['parameter_names']:
                if spec['kind'] == 'free-lora':
                    deltas[name] = FreeLoRADelta(record.tensors['lora_B.'+name], record.tensors['lora_A.'+name], spec['alpha'])
                    continue
                basis = self.engine.store.get_native_tensor_artifact(record.source_artifact_id).artifact
                key = f'{record.source_artifact_id}:{name}'
                if key not in bases:
                    left, right = basis.tensors['left.'+name], basis.tensors['right.'+name]
                    bases[key] = CanonicalBasis(key, name, left.shape[-1], left, right, tensor_digest(left, right))
                rank = spec['rank']
                deltas[name] = SubspaceDelta(key, rank, record.tensors[name].reshape(*record.tensors[name].shape[:-1], rank, rank))
            members.append(StackMember(node, level, scale, deltas))
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
        return {'schema': 'natlang.tune.training-receipt/1', 'input_sha256': self.input_sha256,
                'base_native_identity': dict(self.engine.native_identity), 'recipe': self.recipe.as_json(),
                'step': self.state['step'], 'epoch': self.state['epoch'], 'targets': self.target_names,
                'target_classes': self.target_classes,
                'exposure': {'rows_seen': sum(exposure.values()), 'distinct_occurrences': len(exposure),
                             'by_binding': {binding: sum(count for occurrence, count in exposure.items()
                                                         if any(row['occurrence'] == occurrence and row['binding'] == binding for row in self.rows))
                                            for binding in sorted({row['binding'] for row in self.rows})}},
                'losses': self.state['losses'], **self.manifest_fragment(),
                'soft_states': dict(self.state.get('soft_states', {})),
                'optimizer_states': self.state['optimizer_states']}
