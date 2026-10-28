"""Component tokens: the learned optimizer's one mixture parametrization (owner decision 2026-10-28,
plans/neuralese/LFM3_COMMITTED_DECISIONS_20261028.md item 13).

Each bank component has one learned, untied output row; one more row is the null token ("no component"). At a scope's
decision position (after the scope label and the available components' keys, `writer.Writer`) the writer's final
hidden state ``h`` gives

    z_k = <row_k, h> / T            over the null token and the view's AVAILABLE components (others are masked to
                                    -inf before the softmax: base dropout and eligibility are this mask)
    p   = softmax(z), truncated to its top-p set and renormalized (execution and training use the same truncation)
    g   = gain_max * sigmoid(W_g h + b_g)          one bounded gain per target group (depth quarter x matrix family)
    c_kg = g_g * p_k                               executed coefficient of view slot k in group g (null mass unused)

A component's sign is absorbed by the trainable bank (a component whose useful direction is negative is learned so),
so coefficients are nonnegative. Rows are BF16 masters updated with stochastic rounding (`train.optim.AdamWSR`); a
row is initialized from its component's description encoded in a real rendered request (the hidden state at the end
of its key, scaled so initial logits lie within ``row_init`` of zero), never from a fixed vocabulary token. The gain
readout is FP32, zero weight, bias at ``logit(gain_init)``.

Labels are distributions and gains: an in-bank fit (`fit.fit_in_bank`) optimizes free logits and gain logits through
the exact composition, and the outer trainer distills them by KL (`mixture_kl`) beside the executor objectives,
which reach the logits and gains through the composition. Meta-RL is the exact expected score over the finite token
set (`expected_score_loss`). Collapse is controlled by the temperature and the batch load-balance term (`balance`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError

FORMAT_SCHEMA = 'natlang.lopt.mixture-format/1'
HEAD_SCHEMA = 'natlang.lopt.component-tokens/1'
NULL = 'null'


@dataclass(frozen=True)
class MixtureFormat:
    """The executed law of the component-token distribution, bound into every bank snapshot."""
    gain_max: float = 2.
    gain_init: float = .25
    temperature: float = 1.
    top_p: float = .95

    def __post_init__(self):
        for name in ('gain_max', 'temperature'):
            value = getattr(self, name)
            if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
                raise LearnedOptimizerError(f'mixture {name} must be a positive finite float')
        if not 0 < self.gain_init < 1 or not 0 < self.top_p <= 1:
            raise LearnedOptimizerError('gain_init lies in (0, 1) (a share of gain_max) and top_p in (0, 1]')

    def as_json(self):
        return {'schema': FORMAT_SCHEMA, 'gainMax': self.gain_max.hex(), 'gainInit': float(self.gain_init).hex(),
                'temperature': self.temperature.hex(), 'topP': float(self.top_p).hex()}

    @property
    def id(self):
        return 'nmixf1:' + canonical_json_sha256_hex(self.as_json())

    @classmethod
    def from_json(cls, value):
        if value.get('schema') != FORMAT_SCHEMA:
            raise LearnedOptimizerError('not a component-token mixture format')
        return cls(float.fromhex(value['gainMax']), float.fromhex(value['gainInit']), float.fromhex(value['temperature']),
                   float.fromhex(value['topP']))


def gain_bias(fmt):
    return math.log(fmt.gain_init / (1 - fmt.gain_init))


class MixtureHead(nn.Module):
    """Component rows (row 0: the null token) and the per-group gain readout."""

    def __init__(self, width, groups, components, fmt, *, row_init=4., dtype=torch.bfloat16):
        super().__init__()
        if type(groups) is not int or groups < 1:
            raise LearnedOptimizerError('the mixture head needs at least one target group')
        self.fmt, self.row_init, self.ids = fmt, float(row_init), list(components)
        self.rows = nn.Parameter(torch.zeros(1 + len(self.ids), width, dtype=dtype))
        self.register_buffer('initialized', torch.zeros(1 + len(self.ids), dtype=torch.bool))
        self.initialized[0] = True  # the null row starts at zero: logit 0 against every described component
        self.gain = nn.Linear(width, groups, dtype=torch.float32)
        nn.init.zeros_(self.gain.weight)
        nn.init.constant_(self.gain.bias, gain_bias(fmt))

    def config(self):
        return {'schema': HEAD_SCHEMA, 'width': self.rows.shape[1], 'groups': self.gain.out_features,
                'components': list(self.ids), 'format': self.fmt.id, 'rowInit': self.row_init.hex(),
                'rowDtype': str(self.rows.dtype).removeprefix('torch.')}

    def identity(self):
        values = [t.detach().cpu().double().reshape(-1).tolist() for t in (self.rows, self.gain.weight, self.gain.bias)]
        return 'nmixh1:' + canonical_json_sha256_hex({**self.config(), 'initialized': self.initialized.tolist(),
                                                      'values': [[float(x).hex() for x in v] for v in values]})

    def tensors(self):
        return {'rows': self.rows.detach().cpu().clone(), 'initialized': self.initialized.detach().cpu().float(),
                'gain.weight': self.gain.weight.detach().cpu().clone(), 'gain.bias': self.gain.bias.detach().cpu().clone()}

    def load(self, tensors):
        with torch.no_grad():
            if tensors['rows'].shape != self.rows.shape or tensors['rows'].dtype != self.rows.dtype:
                raise LearnedOptimizerError('saved component rows differ from the bank in shape or dtype')
            self.rows.copy_(tensors['rows'])
            self.initialized.copy_(tensors['initialized'] > 0)
            self.gain.weight.copy_(tensors['gain.weight'])
            self.gain.bias.copy_(tensors['gain.bias'])

    def positions(self, component_ids):
        try:
            return [1 + self.ids.index(component) for component in component_ids]
        except ValueError as error:
            raise LearnedOptimizerError(f'component outside the mixture head: {error}') from None

    def initialize_rows(self, component_ids, key_hidden, decision):
        """Rows of not yet initialized components from their keys' hidden states in this real request, scaled so the
        initial logits against ``decision`` lie within ``row_init`` of zero. Returns the initialized component IDs."""
        done = []
        with torch.no_grad():
            norm = float(decision.float().norm()) or 1.
            for component, hidden in zip(component_ids, key_hidden):
                position = self.positions([component])[0]
                if bool(self.initialized[position]):
                    continue
                direction = hidden.float() / (float(hidden.float().norm()) or 1.)
                self.rows[position] = (direction * self.row_init / norm).to(self.rows.dtype)
                self.initialized[position] = True
                done.append(component)
        return done

    def logits(self, hidden, component_ids, available):
        """(1 + M,) FP32 logits over [null, view slots]; unavailable slots are -inf."""
        rows = self.rows[[0, *self.positions(component_ids)]].float()
        z = rows @ hidden.float() / self.fmt.temperature
        mask = torch.tensor([True, *available], device=z.device)
        return torch.where(mask, z, torch.full_like(z, -math.inf))

    def gains(self, hidden):
        return self.fmt.gain_max * torch.sigmoid(self.gain(hidden.float()))

    def resized(self, component_ids):
        """A head over another component list (bank growth/adoption): rows follow their component IDs; new rows await
        initialization."""
        head = MixtureHead(self.rows.shape[1], self.gain.out_features, component_ids, self.fmt, row_init=self.row_init,
                           dtype=self.rows.dtype)
        with torch.no_grad():
            head.gain.load_state_dict(self.gain.state_dict())
            head.rows[0], head.initialized[0] = self.rows[0], True
            for component in component_ids:
                if component in self.ids:
                    old, new = self.positions([component])[0], head.positions([component])[0]
                    head.rows[new], head.initialized[new] = self.rows[old], self.initialized[old]
        return head


def top_p_keep(probs, top_p):
    """The smallest highest-probability set whose mass reaches ``top_p`` (always the most likely token); a mask."""
    probs = probs.detach()
    order = torch.argsort(probs, descending=True, stable=True)
    before = torch.cumsum(probs[order], 0) - probs[order]
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[order[before < top_p]] = True
    return keep & (probs > 0)


def distribution(logits, top_p, keep=None):
    """``(p, keep)``: the softmax renormalized over the top-p set (``keep`` given: that recorded set)."""
    full = torch.softmax(logits.float(), -1)
    keep = top_p_keep(full, top_p) if keep is None else torch.as_tensor(keep, dtype=torch.bool, device=logits.device)
    kept = torch.where(keep, logits.float(), torch.full_like(logits.float(), -math.inf))
    return torch.softmax(kept, -1), keep


def coefficients(probs, gains):
    """(M, G) executed coefficients ``g_g * p_k`` of the view slots (``probs`` includes the null token first)."""
    return probs[1:, None] * gains[None, :]


def one_hot(size, token):
    """The distribution of a single chosen token (view position ``token``, or `NULL`), length ``1 + size``."""
    probs = torch.zeros(1 + size)
    probs[0 if token == NULL else 1 + int(token)] = 1.
    return probs


def mixture_kl(target, logits):
    """``KL(target || softmax(logits))`` over the untruncated masked student law (labels are truncated fits)."""
    target = torch.as_tensor(target, dtype=torch.float32, device=logits.device)
    logp = torch.log_softmax(logits.float(), -1)
    support = target > 0
    return (target[support] * (target[support].log() - logp[support])).sum()


def gain_loss(target, gains, fmt):
    """Squared error of the gains in their logit coordinates (the readout's own scale)."""
    def logit(g):
        share = (torch.as_tensor(g, dtype=torch.float32, device=gains.device) / fmt.gain_max).clamp(1e-6, 1 - 1e-6)
        return torch.log(share) - torch.log1p(-share)
    return ((logit(gains) - logit(target)) ** 2).mean()


def balance(distributions):
    """Load balance across a batch: ``log n - H(mean p)`` over the distinct tokens (bank component IDs and the null
    token) the batch's distributions cover; zero when usage is uniform. ``distributions``: [(token ids, p)]."""
    totals, count = {}, len(distributions)
    for tokens, probs in distributions:
        for token, value in zip(tokens, probs):
            totals[token] = totals[token] + value if token in totals else value
    if count == 0 or len(totals) < 2:
        return None
    mean = torch.stack(list(totals.values())) / count
    entropy = -(mean[mean > 0] * mean[mean > 0].log()).sum()
    return math.log(len(totals)) - entropy


def expected_score_loss(probs, rewards):
    """``-sum_k p_k (R_k - b)`` with ``b = sum_k p_k R_k`` detached: the exact gradient of the expected score over the
    scored finite token set (the baseline leaves it unchanged and records the centered advantages)."""
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=probs.device)
    baseline = (probs.detach() * rewards).sum()
    return -(probs * (rewards - baseline)).sum()


def receipt(head, view_ids, logits, probs, keep, gains, executed, *, chosen=None):
    hexes = lambda tensor: [float(x).hex() for x in tensor.detach().double().reshape(-1).tolist()]
    return {'schema': HEAD_SCHEMA, 'head': head.identity() if head is not None else None, 'tokens': [NULL, *view_ids],
            'logits': None if logits is None else hexes(logits), 'distribution': hexes(probs),
            'kept': [bool(x) for x in keep.tolist()], 'gains': hexes(gains), 'chosen': chosen,
            'shape': list(executed.shape), 'executed': hexes(executed)}
