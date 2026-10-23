"""Current SVD low-rank weight adapter math for native nnt1 actors."""
from __future__ import annotations

import torch


def singular_bases(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic sign-canonical rank-r factorization basis."""
    if type(rank) is not int or rank < 1 or rank > min(weight.shape[-2:]):
        raise ValueError('native SVD adapter rank must fit the target matrix')
    left_all, _, right_h_all = torch.linalg.svd(weight, full_matrices=False)
    left, right = left_all[..., :rank], right_h_all[..., :rank, :].transpose(-2, -1)
    signs = torch.sign(left.gather(-2, left.abs().argmax(-2, keepdim=True))).clamp_min(0) * 2 - 1
    return (left * signs).contiguous(), (right * signs).contiguous()
