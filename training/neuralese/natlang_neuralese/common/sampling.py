"""One normalized token-mask law for native serving and graph-policy replay."""
from __future__ import annotations
import math
import torch


def masked_log_distribution(logits,temperature,*,allowed=None,banned=None):
    from .stable_softmax import log_softmax
    return log_softmax(masked_logits(logits,temperature,allowed=allowed,banned=banned),dim=-1)


# The sampling law of RL collection and held evaluation: temperature 1 with nucleus truncation (owner 2026-11-13).
# Untruncated T=1 sampling drew a token of probability < 0.0067 once in 55 samples (about 150 per episode) and 34% of
# the policy-gradient mass sat on tokens with p < 0.1 (DECISIONS 2026-11-13). One default, read by every collector
# (grpo.py, exp1_collect.py, held_eval): 1.0 turns truncation off.
RL_TOP_P = 0.95
# `sampling_law.probabilities` of a capture: what a sample's recorded `logp` is. Under top-p the engines record the
# untruncated post-temperature log-probability (the SGLang fork's record, `natlang_post_temperature_logprobs`); the
# draw itself came from the top-p nucleus of that distribution, renormalized.
NORMALIZED_LAW = 'post-processing-normalized'
UNTRUNCATED_LAW = 'post-temperature-untruncated'


def check_top_p(value):
    """A request's top_p as a float in (0, 1] (None: 1.0)."""
    if value is None:
        return 1.
    if type(value) not in (int, float) or not math.isfinite(value) or not 0. < value <= 1.:
        raise ValueError('top_p must be a number in (0, 1]')
    return float(value)


def rl_sampling_controls(top_p=RL_TOP_P, *, capture=False):
    """The chat request controls (`sampling-config.json`, `run_bonsai_queue --chat-request-config`) of an RL or held
    episode: no guidance, the sampling capture when a learner reads the rollouts (`capture`), top-p when below 1."""
    top_p = check_top_p(top_p)
    return {'guidance': False, **({'x_natlang_rl_capture': True, 'x_natlang_rl_top_k': 0} if capture else {}),
            **({'top_p': top_p} if top_p < 1. else {})}


def sampling_law_of(top_p=RL_TOP_P):
    """The law a run records beside its results ({temperature, top_p}): results of different laws are never paired."""
    return {'temperature': 1., 'top_p': check_top_p(top_p)}


def nucleus_keep(sorted_probs, top_p):
    """Mask over probabilities sorted by decreasing value: a token is kept while the mass before it is <= top_p (the
    SGLang sampler's rule, `top_k_top_p_min_p_sampling_from_probs_torch`); the first token is always kept."""
    keep = (torch.cumsum(sorted_probs, -1) - sorted_probs) <= top_p
    keep[..., 0] = True
    return keep


def nucleus_sample(log_probs, top_p, generator=None):
    """One token id drawn from the top-p nucleus of a 1-D normalized log-distribution (renormalized)."""
    probs, order = log_probs.exp().sort(descending=True)
    probs = probs * nucleus_keep(probs, top_p)
    return int(order[torch.multinomial(probs.cpu(), 1, generator=generator)])


def masked_token_logprobs(logits,labels,temperature,*,allowed=None,banned=None,policy=None):
    """Selected-token probabilities under the exact serving mask and stable gradient law."""
    from .stable_softmax import token_logprobs
    return token_logprobs(masked_logits(logits,temperature,allowed=allowed,banned=banned),labels,policy=policy)


def masked_logits(logits,temperature,*,allowed=None,banned=None):
    """Temperature-scaled FP32 logits with masked tokens at -inf before normalization."""

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
    return values


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
DEFAULT_BEHAVIOR_TOP_K = 128


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


# Exact mass receipt (versioned field ``mass_v1`` of a top-k receipt). The float16 log-probabilities lose about
# 2^-11 relative precision each, so 1 - sum(exp(logps)) is not the sampler's tail mass: it can be negative or badly
# wrong when the head holds nearly everything. The actor records the FP32 tail mass summed directly over the tail
# entries (and the FP32 head mass). Receipts without the field keep the reconstructed path (clamped and reported).
TOP_K_MASS_FIELD = 'mass_v1'


def encode_top_k(log_probs, k):
    """Top-k of one normalized 1-D log distribution as a compact JSON receipt (with its exact FP32 head/tail mass)."""
    import base64
    log_probs = log_probs.detach().float()
    values, ids = torch.topk(log_probs, min(k, log_probs.shape[-1]))
    probs = log_probs.exp()
    head = float(probs.gather(-1, ids).sum(dtype=torch.float64))
    tail = float(probs.index_fill(-1, ids, 0.).sum(dtype=torch.float64))
    ids = ids.to('cpu', torch.int32).numpy().astype('<i4')
    logps = values.to('cpu').numpy().astype('<f2')
    return {'ids': base64.b64encode(ids.tobytes()).decode('ascii'),
            'logps': base64.b64encode(logps.tobytes()).decode('ascii'),
            TOP_K_MASS_FIELD: {'head': head, 'tail': tail}}


def top_k_mass(receipt):
    """(head, tail) FP32 masses of a receipt that carries the versioned exact-mass field, else None; validated."""
    mass = receipt.get(TOP_K_MASS_FIELD) if isinstance(receipt, dict) else None
    if mass is None:
        return None
    if (not isinstance(mass, dict) or set(mass) != {'head', 'tail'}
            or any(type(mass[key]) not in (int, float) or isinstance(mass[key], bool) for key in mass)):
        raise ValueError('behavior top-k exact mass must be {head, tail} numbers')
    head, tail = float(mass['head']), float(mass['tail'])
    if not (0. <= head <= 1. + 1e-4 and 0. <= tail <= 1. + 1e-4) or abs(head + tail - 1.) > 1e-3:
        raise ValueError('behavior top-k exact mass is not a probability split')
    return head, tail


def decode_top_k_arrays(receipt):
    """(int64 IDs, float32 log-probs) numpy arrays of one encoded receipt, validated.

    Validation runs in numpy: a learner admission decodes every sampled position's receipt (millions per round),
    where per-receipt torch operator overhead dominated the preparation time."""
    import base64
    import numpy as np
    if not isinstance(receipt, dict) or set(receipt) - {TOP_K_MASS_FIELD} != {'ids', 'logps'}:
        raise ValueError('behavior top-k receipt must contain exactly ids and logps (and optionally its exact mass)')
    try:
        ids = np.frombuffer(base64.b64decode(receipt['ids'], validate=True), dtype='<i4')
        logps = np.frombuffer(base64.b64decode(receipt['logps'], validate=True), dtype='<f2')
    except (ValueError, TypeError) as error:
        raise ValueError('behavior top-k receipt is not valid packed base64') from error
    if not len(ids) or len(ids) != len(logps) or len(ids) > MAX_BEHAVIOR_TOP_K:
        raise ValueError('behavior top-k receipt has inconsistent or empty lengths')
    ids = ids.astype(np.int64)
    logps = logps.astype(np.float32)
    ordered = np.sort(ids)
    if ordered[0] < 0 or (ordered[1:] == ordered[:-1]).any():
        raise ValueError('behavior top-k receipt must name distinct nonnegative token IDs')
    if (np.isnan(logps).any() or (logps == np.inf).any() or (logps > 1e-3).any()
            or (logps[1:] > logps[:-1]).any()):
        raise ValueError('behavior top-k log-probabilities must be nonpositive and sorted')
    return ids, logps


def decode_top_k(receipt):
    """(int64 IDs, float32 log-probs) tensors of one encoded receipt, validated."""
    ids, logps = decode_top_k_arrays(receipt)
    return torch.from_numpy(ids), torch.from_numpy(logps)
