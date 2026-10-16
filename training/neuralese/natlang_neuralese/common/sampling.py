"""One normalized token-mask law for native serving and graph-policy replay."""
from __future__ import annotations
import math
import torch


def masked_log_distribution(logits,temperature,*,allowed=None,banned=None):
    if not math.isfinite(float(temperature)) or temperature<=0:
        raise ValueError('sample temperature must be finite and positive')
    width=logits.shape[-1]
    allowed,banned=token_mask(width,allowed=allowed,banned=banned)
    values=logits.float()/temperature
    if allowed is not None:
        mask=torch.zeros(width,dtype=torch.bool,device=logits.device)
        mask[list(allowed)]=True
        values=values.masked_fill(~mask,float('-inf'))
    elif banned:
        values=values.index_fill(-1,torch.tensor(banned,device=logits.device),float('-inf'))
    return torch.log_softmax(values,dim=-1)


def token_mask(width,*,allowed=None,banned=None):
    """Exact shared mask inventory validation without a vocabulary allocation."""
    def indices(values):
        if values is None:return None
        result=tuple(values)
        if len(set(result))!=len(result) or any(type(value) is not int or not 0<=value<width for value in result):
            raise ValueError('sample mask must uniquely name in-vocabulary token IDs')
        return result
    allowed,banned=indices(allowed),indices(banned)
    if allowed is not None and banned:
        raise ValueError('sample mask must use either allowed or banned token IDs')
    if allowed is not None and not allowed:
        raise ValueError('sample mask has no allowed tokens')
    if banned and len(banned)==width:
        raise ValueError('sample mask excludes every token')
    return allowed,banned
