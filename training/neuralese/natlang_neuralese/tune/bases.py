"""One canonical maximum-rank basis per base matrix; ranks are exact leading slices.

Independent SVD calls are not a nesting contract: members of a stack that sum in
a common basis must name the same basis ID and slice it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..common.hashing import canonical_json_sha256_hex, sha256_hex
from ..model.native_adapter_math import singular_bases
from . import TuneMathError

METHOD = 'weight-svd-sign-canonical-v2'


def basis_id(native_identity, parameter_name, max_rank, shape, method=METHOD):
    return 'nb1:'+canonical_json_sha256_hex({'schema': 'natlang.tune.basis/1', 'native_identity': native_identity,
        'parameter_name': parameter_name, 'max_rank': max_rank, 'shape': list(shape), 'method': method})


def tensor_digest(*tensors):
    return sha256_hex(b''.join(str((tuple(t.shape), str(t.dtype))).encode()+t.detach().cpu().contiguous().numpy().tobytes()
                               for t in tensors))


@dataclass(frozen=True)
class CanonicalBasis:
    basis_id: str
    parameter_name: str
    max_rank: int
    left: torch.Tensor   # (..., rows, R)
    right: torch.Tensor  # (..., cols, R)
    content_sha256: str

    def slice(self, rank):
        """Exact leading-slice views (no copy) of the canonical basis."""
        if type(rank) is not int or not 1 <= rank <= self.max_rank:
            raise TuneMathError(f'rank {rank} is outside basis {self.basis_id} (max {self.max_rank})')
        return self.left[..., :rank], self.right[..., :rank]

    def verify(self):
        if tensor_digest(self.left, self.right) != self.content_sha256:
            raise TuneMathError(f'basis {self.basis_id} content differs from its recorded digest')
        return self


def canonical_basis(native_identity, parameter_name, weight, max_rank):
    """Compute once from the exact base weight in FP32; store FP32."""
    left, right = singular_bases(weight.detach().float(), max_rank)
    return CanonicalBasis(basis_id(native_identity, parameter_name, max_rank, weight.shape), parameter_name, max_rank,
                          left, right, tensor_digest(left, right))


@dataclass(frozen=True)
class BasisSlice:
    basis_id: str
    rank: int


def verify_nested(members, bases):
    """All members must slice one shared basis; returns it. Rejects unrelated bases claimed as nested."""
    ids = {member.basis_id for member in members}
    if len(ids) != 1:
        raise TuneMathError(f'common-basis summation needs one shared basis, got {sorted(ids)}')
    basis = bases.get(next(iter(ids)))
    if basis is None:
        raise TuneMathError('common-basis summation names an unavailable basis')
    basis.verify()
    for member in members:
        basis.slice(member.rank)
    return basis
