"""Stack members from a release's files alone: each runtime stack entry (artifact ID, scale, node) as the trainer's
`StackMember` with its canonical basis views, decoded from the release's `artifacts/*.nnt` (no engine is loaded).
Family training reads member releases through these (`tune.families`).
"""
from __future__ import annotations

from pathlib import Path

from . import TuneMathError
from .bases import CanonicalBasis, tensor_digest
from .parameterization import FreeLoRADelta, SubspaceDelta
from .stack import StackMember


def load_artifacts(directory):
    from ..model.native_artifact import decode_native_tensor_artifact
    artifacts = {}
    for path in sorted(Path(directory).glob('*.nnt')):
        artifact, ident = decode_native_tensor_artifact(path.read_bytes())
        if path.stem != ident:
            raise TuneMathError(f'artifact file {path.name} does not hold its named content')
        artifacts[ident] = artifact
    return artifacts


def members_from_artifacts(artifacts, entries):
    """`StackMember`s and basis views for runtime stack entries (artifact ID, scale, node)."""
    members, bases, shapes = [], {}, {}
    for entry in entries:
        record = artifacts.get(entry['id'])
        if record is None or record.kind != 'adapter-coefficients':
            raise TuneMathError(f"stack entry {entry['id']} is not a coefficient artifact of this release")
        spec = record.provenance['adapter_spec']
        deltas = {}
        for name in spec['parameter_names']:
            if spec['kind'] == 'free-lora':
                B, A = record.tensors['lora_B.'+name], record.tensors['lora_A.'+name]
                deltas[name] = FreeLoRADelta(B, A, spec['alpha'])
                shapes[name] = (*B.shape[:-1], A.shape[-1])
                continue
            basis = artifacts.get(record.source_artifact_id)
            if basis is None:
                raise TuneMathError(f"coefficients {entry['id']} lack their basis artifact in this release")
            key = f'{record.source_artifact_id}:{name}'
            left, right = basis.tensors['left.'+name], basis.tensors['right.'+name]
            if key not in bases:
                bases[key] = CanonicalBasis(key, name, left.shape[-1], left, right, tensor_digest(left, right))
            rank = spec['rank']
            core = record.tensors[name]
            deltas[name] = SubspaceDelta(key, rank, core.reshape(*core.shape[:-1], rank, rank))
            shapes[name] = (*left.shape[:-1], right.shape[-2])
        level = entry['node'].split(':', 1)[0]  # family:, tree:, folder:, lambda: scope node IDs
        members.append(StackMember(entry['node'], level, float(entry['scale']), deltas))
    return members, bases, shapes
