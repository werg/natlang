"""Shared basis, component bank and learned keys (plan "A bank large enough to move the needle", "A component bank
with learned soft descriptions", "Basis fitting", "Stable coordinates").

Per target matrix ``m`` one shared basis pair ``U_m, V_m`` of rank ``R`` (orthonormal columns, FP32). Component ``k``
has a low-rank core ``C_km = P_km Q_km^T`` inside that subspace (``P, Q`` of shape ``[..., R, r_c]``; leading expert
axes stay explicit). A scope's executed core is

    C_sm = sum_k c_{s,k,g(m)} C_km,        Delta W_sm = U_m C_sm V_m^T

with ``c`` the executed coefficients (`head.executed`) and ``g(m)`` the target group of ``m`` (depth quarter x matrix
family). The composed core is materialized as an ordinary native ``xs`` adapter that slices the bank's basis
artifact (`bank-basis-slice-v1`), so the existing execution, gradient, serving and export paths run it unchanged.

Components are normalized: ``sum_m q_m ||C_km||_F^2 = 1`` with ``q_m = 1/(numel_m ms(W_m))`` (energy relative to the
base weight's mean square), so integer strengths
are comparable and rescaling a component cannot distort sparsity or precision. A snapshot's identity binds the basis,
ordered component IDs and tensors, keys, eligibility and number format; any change is a new snapshot, and an old
weight array never resolves against it (`views.validate_array`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

import torch

from ...common.hashing import canonical_json_sha256_hex
from ..bases import tensor_digest
from ..gates import factor_sq_norm
from . import LearnedOptimizerError
from .numbers import NumberFormat

BASIS_METHOD = 'bank-shared-basis-v1'
SLICE_METHOD = 'bank-basis-slice-v1'
BANK_SCHEMA = 'natlang.lopt.component-bank/1'
FAMILY_OF_KIND = {'attention': 'attention', 'conv-projection': 'convolution', 'mlp': 'mlp', 'experts': 'experts', 'router': 'router'}
_LAYER = re.compile(r'\.layers\.(\d+)\.')


# -- target groups ------------------------------------------------------------------------------------------------
def target_groups(names, kinds, num_layers, quarters=4):
    """``(ordered group labels, {name: group index})``: depth quarter by matrix family (12 groups on LFM2 and LFM3:
    attention incl. LFM3's per-head gate, convolution projections, feed-forward)."""
    labels, index = [], {}
    for name in sorted(names):
        match = _LAYER.search(name)
        if match is None or name not in kinds:
            raise LearnedOptimizerError(f'target {name} has no layer index or matrix family')
        quarter = min(quarters - 1, int(match.group(1)) * quarters // num_layers)
        label = f'q{quarter}:{FAMILY_OF_KIND[kinds[name]]}'
        if label not in labels:
            labels.append(label)
    labels.sort()
    for name in names:
        quarter = min(quarters - 1, int(_LAYER.search(name).group(1)) * quarters // num_layers)
        index[name] = labels.index(f'q{quarter}:{FAMILY_OF_KIND[kinds[name]]}')
    return labels, index


# -- basis --------------------------------------------------------------------------------------------------------
def orthonormalize(matrix):
    """``(Q, R)`` with orthonormal Q and a sign-canonical (positive-diagonal) R: ``matrix = Q R``."""
    q, r = torch.linalg.qr(matrix.float())
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs[..., None, :], r * signs[..., :, None]


def transform_core(core, r_left, r_right):
    """Re-express a core after ``U = Q_u R_u``, ``V = Q_v R_v``: ``C' = R_u C R_v^T`` (same subspace)."""
    return r_left @ core @ r_right.transpose(-2, -1)


@dataclass
class SharedBasis:
    """One rank-``R`` basis pair per target matrix of one exact base."""
    native_identity: dict
    rank: int
    left: dict    # name -> (..., rows, R)
    right: dict   # name -> (..., cols, R)
    method: dict  # how it was built: {'kind': ..., 'sources': [...]} (provenance, part of the identity)

    @property
    def names(self):
        return sorted(self.left)

    def spec(self):
        return {'schema': 'natlang.native-adapter/1', 'kind': 'bank-basis', 'bases': BASIS_METHOD, 'max_rank': self.rank,
                'parameter_names': self.names, 'method': self.method}

    @property
    def content_sha256(self):
        return tensor_digest(*[t for name in self.names for t in (self.left[name], self.right[name])])

    def check(self, tolerance=1e-4):
        for name in self.names:
            for side in (self.left[name], self.right[name]):
                if side.dtype != torch.float32 or side.shape[-1] != self.rank:
                    raise LearnedOptimizerError(f'basis of {name} must be FP32 with rank {self.rank}')
                gram = side.transpose(-2, -1) @ side
                eye = torch.eye(self.rank, dtype=gram.dtype, device=gram.device)
                if float((gram - eye).abs().max()) > tolerance:
                    raise LearnedOptimizerError(f'basis of {name} has no orthonormal columns')
        return self

    def to_artifact(self, engine, owner=None):
        from ...model.native_artifact import NativeTensorArtifact
        if self.native_identity != engine.native_identity:
            raise LearnedOptimizerError('the basis belongs to another exact base')
        tensors = {}
        for name in self.names:
            tensors['left.' + name], tensors['right.' + name] = self.left[name].contiguous().cpu(), self.right[name].contiguous().cpu()
        return engine.store.put_native_tensor_artifact(NativeTensorArtifact(
            'adapter-basis', engine.native_identity, tensors, provenance={'adapter_spec': self.spec()}), owner)

    @classmethod
    def from_artifact(cls, engine, artifact_id):
        record = engine.store.get_native_tensor_artifact(artifact_id)
        spec = (record.artifact.provenance or {}).get('adapter_spec') if record else None
        if record is None or record.artifact.kind != 'adapter-basis' or not spec or spec.get('bases') != BASIS_METHOD:
            raise LearnedOptimizerError(f'{artifact_id} is not a bank basis artifact')
        tensors = record.artifact.tensors
        return cls(record.artifact.native_identity, spec['max_rank'],
                   {name: tensors['left.' + name] for name in spec['parameter_names']},
                   {name: tensors['right.' + name] for name in spec['parameter_names']}, spec['method'])


def weight_svd_basis(engine, names, rank):
    """The canonical weight-SVD directions as a bank basis (one ingredient of `subspace.pooled_basis`)."""
    from ...model.native_adapter_math import singular_bases
    from ...model.native_adapters import target_parameters
    available = target_parameters(engine.shared_model)
    left, right = {}, {}
    with torch.no_grad():
        for name in names:
            left[name], right[name] = (t.cpu() for t in singular_bases(available[name].detach().float(), rank))
    return SharedBasis(dict(engine.native_identity), rank, left, right, {'kind': 'weight-svd', 'sources': []})


# -- components ---------------------------------------------------------------------------------------------------
@dataclass
class Component:
    id: str
    factors: dict              # name -> (P (..., R, r_c), Q (..., R, r_c)); absent matrices contribute zero
    key: str | None            # nzs1_ state of Neuralese<ComponentDescription>
    description: str           # the key's source text (provenance; re-initialization source)
    source: dict               # {'kind': 'gradient-subspace'|'source-fit'|'free-tune'|'residual'|'planted', ...}
    levels: tuple = ('tree', 'folder', 'lambda')   # scope levels where the component is eligible

    def core(self, name):
        if name not in self.factors:
            return None
        P, Q = self.factors[name]
        return P @ Q.transpose(-2, -1)

    def as_json(self):
        return {'id': self.id, 'key': self.key, 'description': self.description, 'source': self.source,
                'levels': list(self.levels), 'matrices': sorted(self.factors)}


def component_id(source, factors):
    return 'ncmp1:' + canonical_json_sha256_hex({'source': source, 'tensors': tensor_digest(
        *[t for name in sorted(factors) for t in factors[name]])})


def core_norms(component, q):
    """Per-matrix ``q_m ||C_km||_F^2`` from the factors, without forming the core."""
    return {name: q[name] * factor_sq_norm(P, Q.transpose(-2, -1)) for name, (P, Q) in component.factors.items()}


def normalize(component, q, *, floor=1e-12):
    """Scale the component so ``sum_m q_m ||C_km||^2 = 1``; returns ``(component, scale)``. A near-zero component is
    rejected: it has no direction to carry."""
    total = float(sum(core_norms(component, q).values(), torch.zeros(())))
    if total < floor:
        raise LearnedOptimizerError(f'component {component.id} has no measurable update; record it as absent')
    scale = 1. / math.sqrt(total)
    root = math.sqrt(scale)
    factors = {name: (P * root, Q * root) for name, (P, Q) in component.factors.items()}
    return replace(component, factors=factors, id=component_id(component.source, factors)), scale


def factor_core(core, rank):
    """Best rank-``r_c`` factorization ``P Q^T`` of a core (truncated SVD, singular values split evenly)."""
    u, s, vh = torch.linalg.svd(core.float(), full_matrices=False)
    root = s[..., :rank].sqrt()
    return u[..., :rank] * root[..., None, :], vh[..., :rank, :].transpose(-2, -1) * root[..., None, :]


# -- the bank -----------------------------------------------------------------------------------------------------
@dataclass
class ComponentBank:
    basis: SharedBasis
    basis_artifact: str
    components: list
    number_format: NumberFormat
    group_labels: list
    group_of: dict             # matrix name -> group index
    q: dict                    # matrix name -> inverse element count
    parent: str | None = None
    history: list = field(default_factory=list)
    _offsets: dict = field(default_factory=dict, repr=False, compare=False)

    def manifest(self):
        return {'schema': BANK_SCHEMA, 'native_identity': self.basis.native_identity, 'basis': self.basis_artifact,
                'basis_sha256': self.basis.content_sha256, 'rank': self.basis.rank,
                'components': [component.as_json() for component in self.components],
                'component_tensors': tensor_digest(*[t for c in self.components for name in sorted(c.factors) for t in c.factors[name]]),
                'number_format': self.number_format.as_json(), 'groups': self.group_labels, 'group_of': self.group_of,
                'parent': self.parent}

    @property
    def snapshot(self):
        return 'nbank1:' + canonical_json_sha256_hex(self.manifest())

    @property
    def ids(self):
        return [component.id for component in self.components]

    def index(self, component_id):
        return self.ids.index(component_id)

    def group_norms(self):
        """(K, G) per-group update energy ``sum_{m in g} q_m ||C_km||^2`` (the head's modulation shares)."""
        out = torch.zeros(len(self.components), len(self.group_labels))
        for k, component in enumerate(self.components):
            for name, value in core_norms(component, self.q).items():
                out[k, self.group_of[name]] += float(value)
        return out

    def composed_cores(self, slots, coefficients, *, factors=None, offset=None):
        """``{m: C_m}`` for view slots ``slots`` (component indices) and executed coefficients (len(slots), G).

        ``factors`` (optional) maps component index -> {m: (P, Q)} live leaves; the result is then differentiable in
        them and in ``coefficients``. ``offset`` (optional, from `adapter_cores`) is the incoming adapter's cores: the
        mixture then revises an already adapted scope (a tuned program, a partial tune or another program's bundle)."""
        if coefficients.shape != (len(slots), len(self.group_labels)):
            raise LearnedOptimizerError('executed coefficients must be (slots, groups)')
        R = self.basis.rank
        cores = {}
        for name in self.basis.names:
            group = self.group_of[name]
            total = None
            for row, k in enumerate(slots):
                pair = (factors or {}).get(k, self.components[k].factors).get(name)
                if pair is None:
                    continue
                term = coefficients[row, group] * (pair[0] @ pair[1].transpose(-2, -1))
                total = term if total is None else total + term
            if offset is not None and name in offset:
                base = offset[name].to(coefficients)
                total = base if total is None else total + base
            if total is None:
                shape = (*self.basis.left[name].shape[:-2], R, R)
                total = coefficients.new_zeros(shape)
            cores[name] = total
        return cores

    def adapter_spec(self):
        return {'schema': 'natlang.native-adapter/1', 'kind': 'xs', 'rank': self.basis.rank, 'seed': 0,
                'parameter_names': self.basis.names, 'bases': SLICE_METHOD, 'basis_spec': self.basis.spec()}

    def adapter_tensors(self, cores):
        return {name: core.reshape(*core.shape[:-2], -1).float() for name, core in cores.items()}

    def compose_adapter(self, engine, cores, owner=None, provenance=None):
        """Store the composed cores as an executable native ``xs`` adapter slicing the bank basis."""
        from ...model.native_artifact import NativeTensorArtifact
        tensors = {name: value.detach().cpu().contiguous() for name, value in self.adapter_tensors(cores).items()}
        return engine.store.put_native_tensor_artifact(NativeTensorArtifact(
            'adapter-coefficients', engine.native_identity, tensors, self.basis_artifact,
            {'adapter_spec': self.adapter_spec(), 'bank': {'snapshot': self.snapshot, **(provenance or {})}}), owner)

    def adapter_cores(self, engine, adapter_id):
        """``{m: (..., R, R)}`` cores of any ``xs`` adapter slicing this bank's basis (a tuner node of rank ``r <= R``
        is its leading block, zero elsewhere: exact). Adapters on another basis are not coordinates of this bank."""
        record = engine.store.get_native_tensor_artifact(adapter_id)
        spec = (record.artifact.provenance or {}).get('adapter_spec') if record else None
        if (spec is None or spec.get('kind') != 'xs' or spec.get('bases') != SLICE_METHOD
                or record.artifact.source_artifact_id != self.basis_artifact):
            raise LearnedOptimizerError(f'adapter {adapter_id} does not slice this bank\'s basis; project it instead')
        R, r = self.basis.rank, spec['rank']
        cores = {}
        for name in spec['parameter_names']:
            value = record.artifact.tensors[name]
            block = value.reshape(*value.shape[:-1], r, r).float()
            core = block.new_zeros((*block.shape[:-2], R, R))
            core[..., :r, :r] = block
            cores[name] = core
        return cores

    def incoming(self, engine, context, scope):
        """The incoming adapter cores of ``scope`` in a target/context (``incoming_adapters``), or None (zero)."""
        adapter = (context.get('incoming_adapters') or {}).get(scope)
        if adapter is None:
            return None
        if adapter not in self._offsets:
            self._offsets[adapter] = self.adapter_cores(engine, adapter)
        return self._offsets[adapter]

    def dense_update(self, name, core):
        return self.basis.left[name] @ core @ self.basis.right[name].transpose(-2, -1)

    # -- projections of observed updates ------------------------------------------------------------------------
    def project_factors(self, factors):
        """Basis coordinates of an observed effective update given as ``{m: (B, A)}`` (``Delta W = B A``):
        ``C* = (U^T B)(A V)``, and per-matrix energies (observed, captured) for residual accounting."""
        cores, energy = {}, {}
        for name, (B, A) in factors.items():
            if name not in self.basis.left:
                energy[name] = (float(factor_sq_norm(B, A)), 0.)
                continue
            U, V = self.basis.left[name].to(B), self.basis.right[name].to(A)
            core = (U.transpose(-2, -1) @ B) @ (A @ V)
            cores[name] = core
            energy[name] = (float(factor_sq_norm(B, A)), float((core ** 2).sum()))
        return cores, energy

    def fit_mixture(self, cores, slots=None, *, ridge=1e-6):
        """Least-squares single coefficient per slot: ``min sum_m q_m ||C*_m - sum_k a_k C_km||^2 + ridge ||a||^2``.
        Returns ``(a, residual energy, target energy)``. Weight-space projection only: it initializes and grows the
        bank; mixture-token labels come from behavioral in-bank fits (`fit.py`)."""
        slots = list(range(len(self.components))) if slots is None else list(slots)
        K = len(slots)
        gram, rhs, target = torch.zeros(K, K, dtype=torch.float64), torch.zeros(K, dtype=torch.float64), 0.
        built = {k: {name: self.components[k].core(name) for name in self.components[k].factors} for k in slots}
        for name, value in cores.items():
            qm = self.q[name]
            target += qm * float((value.double() ** 2).sum())
            parts = [built[k].get(name) for k in slots]
            for i, a in enumerate(parts):
                if a is None:
                    continue
                rhs[i] += qm * float((a.double() * value.double()).sum())
                for j, b in enumerate(parts):
                    if b is not None:
                        gram[i, j] += qm * float((a.double() * b.double()).sum())
        coefficients = torch.linalg.solve(gram + ridge * torch.eye(K, dtype=torch.float64), rhs) if K else rhs
        residual = target - 2 * float(coefficients @ rhs) + float(coefficients @ gram @ coefficients)
        return coefficients.float(), max(residual, 0.), target

    def with_components(self, components, *, reason):
        """A new snapshot with a changed component list; the old snapshot is its recorded parent."""
        return replace(self, components=list(components), parent=self.snapshot,
                       history=[*self.history, {'parent': self.snapshot, 'reason': reason}])


def new_bank(engine, basis, number_format, *, quarters=4, owner=None, components=()):
    """Store the basis and return an empty (or seeded) bank over the engine's selected targets."""
    from ...model.native_adapters import target_parameters
    from ..targets import module_map
    available = target_parameters(engine.shared_model)
    _, kinds = module_map(sorted(available))
    if set(basis.names) - set(kinds):
        raise LearnedOptimizerError('the basis names matrices outside the student target table')
    layers = engine.shared_model.port.num_layers
    labels, group_of = target_groups(basis.names, kinds, layers, quarters)
    # Relative energy: q_m ||C_m||^2 is the update's mean-square entry relative to the base weight's, so a unit
    # coefficient on a normalized component is an update of the weights' own RMS spread over its matrices.
    q = {name: 1. / (available[name].numel() * float(available[name].detach().float().pow(2).mean())) for name in basis.names}
    record = basis.check().to_artifact(engine, owner)
    return ComponentBank(basis, record.id, list(components), number_format, labels, group_of, q)


# -- persistence --------------------------------------------------------------------------------------------------
def save_bank(engine, bank, owner=None):
    """Persist component tensors as one parameter-set artifact; returns the manifest with its tensor reference."""
    from ...model.native_artifact import NativeTensorArtifact
    tensors = {}
    for k, component in enumerate(bank.components):
        for name, (P, Q) in component.factors.items():
            tensors[f'{k}.P.{name}'], tensors[f'{k}.Q.{name}'] = P.detach().cpu().contiguous(), Q.detach().cpu().contiguous()
    manifest = {**bank.manifest(), 'snapshot': bank.snapshot, 'q': {name: float(v).hex() for name, v in bank.q.items()},
                'history': bank.history}
    if tensors:
        record = engine.store.put_native_tensor_artifact(NativeTensorArtifact(
            'parameter-set', engine.native_identity, tensors, bank.basis_artifact,
            {'scope': BANK_SCHEMA, 'snapshot': bank.snapshot}), owner)
        manifest['tensors'] = record.id
    else:
        manifest['tensors'] = None
    return manifest


def load_bank(engine, manifest):
    basis = SharedBasis.from_artifact(engine, manifest['basis'])
    tensors = engine.store.get_native_tensor_artifact(manifest['tensors']).artifact.tensors if manifest['tensors'] else {}
    components = []
    for k, item in enumerate(manifest['components']):
        factors = {name: (tensors[f'{k}.P.{name}'], tensors[f'{k}.Q.{name}']) for name in item['matrices']}
        components.append(Component(item['id'], factors, item['key'], item['description'], item['source'], tuple(item['levels'])))
    bank = ComponentBank(basis, manifest['basis'], components, NumberFormat.from_json(manifest['number_format']),
                         list(manifest['groups']), dict(manifest['group_of']),
                         {name: float.fromhex(v) for name, v in manifest['q'].items()}, manifest['parent'],
                         list(manifest.get('history', [])))
    if bank.snapshot != manifest['snapshot']:
        raise LearnedOptimizerError('restored bank differs from its recorded snapshot')
    return bank
