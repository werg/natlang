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


# Behaviour top-k receipts (score-centered policy gradients, arXiv 2609.20807).
# At each sampled position the sampler records the k most probable tokens of the
# exact distribution it drew from (temperature and token masks applied; the
# native engine has no further truncation sampler such as top-p) as packed
# little-endian int32 IDs and float16 log-probabilities, by decreasing
# probability. 6 bytes per entry before base64.
BEHAVIOR_TOP_K_SCHEMA = 'natlang.behavior-top-k/1'
BEHAVIOR_TOP_K_ENCODING = {'ids': 'int32-le-base64', 'logps': 'float16-le-base64'}
MAX_BEHAVIOR_TOP_K = 4096


def behavior_top_k_declaration(k, *, probabilities='post-processing-normalized'):
    if type(k) is not int or not 1 <= k <= MAX_BEHAVIOR_TOP_K:
        raise ValueError(f'behavior top-k must be an integer in [1, {MAX_BEHAVIOR_TOP_K}]')
    return {'schema': BEHAVIOR_TOP_K_SCHEMA, 'k': k, 'distribution': probabilities, **BEHAVIOR_TOP_K_ENCODING}


def validate_behavior_top_k_declaration(value):
    if (not isinstance(value, dict) or value.get('schema') != BEHAVIOR_TOP_K_SCHEMA
            or value != behavior_top_k_declaration(value.get('k'), probabilities=value.get('distribution'))
            or value['distribution'] != 'post-processing-normalized'):
        raise ValueError('behavior top-k declaration is not the exact supported receipt format')
    return value


def encode_top_k(log_probs, k):
    """Top-k of one normalized 1-D log distribution as a compact JSON receipt."""
    import base64
    values, ids = torch.topk(log_probs.detach().float(), min(k, log_probs.shape[-1]))
    ids = ids.to('cpu', torch.int32).numpy().astype('<i4')
    logps = values.to('cpu').numpy().astype('<f2')
    return {'ids': base64.b64encode(ids.tobytes()).decode('ascii'),
            'logps': base64.b64encode(logps.tobytes()).decode('ascii')}


def decode_top_k(receipt):
    """(int64 IDs, float32 log-probs) of one encoded receipt, validated."""
    import base64
    import numpy as np
    if not isinstance(receipt, dict) or set(receipt) != {'ids', 'logps'}:
        raise ValueError('behavior top-k receipt must contain exactly ids and logps')
    try:
        ids = np.frombuffer(base64.b64decode(receipt['ids'], validate=True), dtype='<i4')
        logps = np.frombuffer(base64.b64decode(receipt['logps'], validate=True), dtype='<f2')
    except (ValueError, TypeError) as error:
        raise ValueError('behavior top-k receipt is not valid packed base64') from error
    if not len(ids) or len(ids) != len(logps) or len(ids) > MAX_BEHAVIOR_TOP_K:
        raise ValueError('behavior top-k receipt has inconsistent or empty lengths')
    ids = torch.from_numpy(ids.astype(np.int64))
    logps = torch.from_numpy(logps.astype(np.float32))
    if (ids < 0).any() or len(set(ids.tolist())) != len(ids):
        raise ValueError('behavior top-k receipt must name distinct nonnegative token IDs')
    if (torch.isnan(logps).any() or (logps == float('inf')).any() or (logps > 1e-3).any()
            or (logps[1:] > logps[:-1]).any()):
        raise ValueError('behavior top-k log-probabilities must be nonpositive and sorted')
    return ids, logps
