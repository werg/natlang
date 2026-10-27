"""The weight-array number format: single-digit signed integers on a recorded grid (plan "Weight arrays").

Both students spell ``[ 3, 0, -2]`` as ``[``, then per value a sign-bearing space token (``Ġ`` or ``Ġ-``) and one
digit token, separated by ``,``, then ``]``. Each available slot is therefore a binary sign choice followed by a
magnitude choice; negative zero is excluded by the grammar. Brackets, separators and every unavailable slot's
``Ġ0`` are forced tokens with no sampling score. The executed value of integer ``n`` is ``step * n`` (refined by the
coefficient head, `head.py`).

`token-expectation-v1` is exact: it needs the digit distribution under both sign branches, which costs one extra
decoding step for the branch not taken. When a caller cannot supply that branch it must use the declared
`token-expectation-realized-magnitude-v1` approximation, which reuses the realized branch's magnitude distribution
for the other sign and is recorded as such.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError

FORMAT_SCHEMA = 'natlang.lopt.number-format/1'
EXACT_ESTIMATOR = 'token-expectation-v1'
APPROX_ESTIMATOR = 'token-expectation-realized-magnitude-v1'


@dataclass(frozen=True)
class NumberFormat:
    """The number-format contract of one bank snapshot: integers in [-max_int, max_int] times ``step``."""
    step: float
    max_int: int = 9
    grid: str = 'signed-int-grid-v1'

    def __post_init__(self):
        if type(self.step) is not float or not math.isfinite(self.step) or self.step <= 0:
            raise LearnedOptimizerError('the number-format step must be a positive finite float')
        if type(self.max_int) is not int or not 1 <= self.max_int <= 9 or self.grid != 'signed-int-grid-v1':
            raise LearnedOptimizerError('the signed integer grid holds single digits: max_int in [1, 9]')

    def as_json(self):
        return {'schema': FORMAT_SCHEMA, 'grid': self.grid, 'step_hex': self.step.hex(), 'max_int': self.max_int}

    @property
    def id(self):
        return 'nfmt1:' + canonical_json_sha256_hex(self.as_json())

    @classmethod
    def from_json(cls, value):
        if not isinstance(value, dict) or value.get('schema') != FORMAT_SCHEMA:
            raise LearnedOptimizerError('not a number-format contract')
        return cls(float.fromhex(value['step_hex']), value['max_int'], value['grid'])

    @property
    def values(self):
        """Integer support in a fixed order, -max .. max."""
        return tuple(range(-self.max_int, self.max_int + 1))


@dataclass(frozen=True)
class GridTokens:
    """Token IDs of the grid's spelling in one tokenizer, verified from its actual encoder."""
    open: int
    close: int
    separator: int
    positive: int
    negative: int
    digits: tuple

    @classmethod
    def from_encoder(cls, encode, max_int=9):
        """``encode(text) -> list[int]`` without special tokens. Rejects a tokenizer whose spelling differs."""
        single = lambda text: _one(encode(text), text)
        open_, close, separator = single('['), single(']'), single(',')
        pairs = {value: encode(f' {value}') for value in range(-max_int, max_int + 1)}
        if any(len(ids) != 2 for ids in pairs.values()):
            raise LearnedOptimizerError('this tokenizer does not spell each grid value as a sign token and one digit')
        positive, negative = pairs[0][0], pairs[-1][0]
        digits = tuple(pairs[value][1] for value in range(max_int + 1))
        for value, ids in pairs.items():
            if ids != [positive if value >= 0 else negative, digits[abs(value)]]:
                raise LearnedOptimizerError(f'grid value {value} has an inconsistent spelling {ids}')
        if len(set(digits)) != len(digits) or positive == negative:
            raise LearnedOptimizerError('grid digit or sign tokens collide')
        tokens = cls(open_, close, separator, positive, negative, digits)
        # The whole template must tokenize as the concatenation of its parts.
        if encode('[ 3, 0, -2]') != tokens.spell([3, 0, -2]):
            raise LearnedOptimizerError('this tokenizer merges grid template tokens; the forced template is not exact')
        return tokens

    def as_json(self):
        return {'open': self.open, 'close': self.close, 'separator': self.separator, 'positive': self.positive,
                'negative': self.negative, 'digits': list(self.digits)}

    def spell(self, values):
        out = [self.open]
        for index, value in enumerate(values):
            if index:
                out.append(self.separator)
            out.extend(self.slot(value))
        out.append(self.close)
        return out

    def slot(self, value):
        if type(value) is not int or abs(value) >= len(self.digits):
            raise LearnedOptimizerError(f'{value!r} is not a grid integer')
        return [self.positive if value >= 0 else self.negative, self.digits[abs(value)]]


def _one(ids, text):
    if len(ids) != 1:
        raise LearnedOptimizerError(f'{text!r} is not a single token in this tokenizer')
    return ids[0]


def text(values):
    """The visible JSON spelling, e.g. ``[ 3, 0, -2]``; parses as an ordinary ``number[]``."""
    return '[' + ','.join(f' {int(value)}' for value in values) + ']'


def layout(tokens, values, available):
    """The forced template of one array as ``[(token, kind, slot)]`` with kind in open/separator/close/forced-sign/
    forced-digit/sign/digit. Unavailable slots are forced zeros; ``values`` must already be zero there."""
    if len(values) != len(available):
        raise LearnedOptimizerError('values and availability differ in length')
    items = [(tokens.open, 'open', None)]
    for slot, (value, free) in enumerate(zip(values, available)):
        if slot:
            items.append((tokens.separator, 'separator', None))
        if not free and value != 0:
            raise LearnedOptimizerError(f'unavailable slot {slot} must be zero')
        sign, digit = tokens.slot(value)
        items.append((sign, 'sign' if free else 'forced-sign', slot))
        items.append((digit, 'digit' if free else 'forced-digit', slot))
    items.append((tokens.close, 'close', None))
    return items


def parse(tokens, ids, available):
    """Exact parse of generated IDs against the forced grammar; any deviation is a recorded contract failure."""
    expected_length = 2 + 3 * len(available) - (1 if available else 0)
    if len(ids) != expected_length:
        raise LearnedOptimizerError(f'weight array has {len(ids)} tokens; the template for {len(available)} slots has {expected_length}')
    values, position = [], 1
    if ids[0] != tokens.open or ids[-1] != tokens.close:
        raise LearnedOptimizerError('weight array must open with [ and close with ]')
    for slot, free in enumerate(available):
        if slot:
            if ids[position] != tokens.separator:
                raise LearnedOptimizerError(f'slot {slot} is not preceded by a separator')
            position += 1
        sign, digit = ids[position], ids[position + 1]
        position += 2
        if sign not in (tokens.positive, tokens.negative) or digit not in tokens.digits:
            raise LearnedOptimizerError(f'slot {slot} is not a sign token followed by one digit')
        magnitude = tokens.digits.index(digit)
        if sign == tokens.negative and magnitude == 0:
            raise LearnedOptimizerError(f'slot {slot} spells negative zero')
        value = -magnitude if sign == tokens.negative else magnitude
        if not free and value != 0:
            raise LearnedOptimizerError(f'unavailable slot {slot} returned {value}; it must be zero')
        values.append(value)
    return values


# -- grammar-restricted distributions --------------------------------------------------------------------------
def sign_logprobs(tokens, logits):
    """log p(+), log p(-) over the two sign tokens at a slot's decision position."""
    pair = torch.stack([logits[..., tokens.positive], logits[..., tokens.negative]], -1).float()
    return torch.log_softmax(pair, -1)


def magnitude_logprobs(tokens, logits, *, negative, max_int):
    """log p(|n| = m | sign) over the grammar's digits (0..max for +, 1..max for -), as a length max+1 vector with
    -inf at excluded magnitudes."""
    index = torch.tensor(tokens.digits[:max_int + 1], device=logits.device)
    chosen = logits[..., index].float()
    if negative:
        chosen = chosen.clone()
        chosen[..., 0] = -math.inf
    return torch.log_softmax(chosen, -1)


def value_logprobs(tokens, fmt, sign_logits, positive_digit_logits, negative_digit_logits):
    """log p(n) over ``fmt.values`` (-max..max) from the sign logits and the digit logits under each branch."""
    signs = sign_logprobs(tokens, sign_logits)
    pos = magnitude_logprobs(tokens, positive_digit_logits, negative=False, max_int=fmt.max_int) + signs[..., :1]
    neg = magnitude_logprobs(tokens, negative_digit_logits, negative=True, max_int=fmt.max_int) + signs[..., 1:]
    return torch.cat([neg[..., 1:].flip(-1), pos], -1)


def expectation(tokens, fmt, sign_logits, realized_digit_logits, realized_negative, other_digit_logits=None):
    """``(E[n], p(n>0) - p(n<0), estimator)``. With ``other_digit_logits`` (the branch not taken) the result is the
    exact `token-expectation-v1`; without it, the declared realized-magnitude approximation."""
    other = other_digit_logits if other_digit_logits is not None else realized_digit_logits
    pos, neg = (other, realized_digit_logits) if realized_negative else (realized_digit_logits, other)
    logp = value_logprobs(tokens, fmt, sign_logits, pos, neg)
    probs = logp.exp()
    support = torch.tensor(fmt.values, device=probs.device, dtype=probs.dtype)
    mean = (probs * support).sum(-1)
    signed = probs[..., fmt.max_int + 1:].sum(-1) - probs[..., :fmt.max_int].sum(-1)
    return mean, signed, (EXACT_ESTIMATOR if other_digit_logits is not None else APPROX_ESTIMATOR)


def quantize(value, fmt):
    """A fitted continuous coefficient as ``(n, rho, saturated)`` with ``value ~ step*sign(n)*(|n|+rho)``,
    ``rho`` in [-1/2, 1/2). A coefficient below half a step rounds to an exact zero (``rho = 0``)."""
    if not math.isfinite(value):
        raise LearnedOptimizerError('cannot quantize a nonfinite coefficient')
    scaled = abs(value) / fmt.step
    magnitude = min(int(math.floor(scaled + .5)), fmt.max_int)
    saturated = scaled > fmt.max_int + .5
    if magnitude == 0:
        return 0, 0., False
    rho = max(-.5, min(scaled - magnitude, .5 - 1e-9))
    return (magnitude if value > 0 else -magnitude), rho, saturated


def grid_step_for(coefficients, max_int=9, quantile=.95):
    """A grid step covering fitted coefficients with rare saturation: the ``quantile`` of their magnitudes maps to
    ``max_int``. Recorded as a new number-format contract; returns ``(step, saturation rate at that step)``."""
    values = sorted(abs(float(value)) for value in coefficients if float(value) != 0)
    if not values:
        raise LearnedOptimizerError('no nonzero fitted coefficients to size the grid step')
    step = values[min(len(values) - 1, int(quantile * len(values)))] / max_int
    saturated = sum(1 for value in values if value / step > max_int + .5) / len(values)
    return step, saturated
