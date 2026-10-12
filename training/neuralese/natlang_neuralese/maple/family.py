"""The nested family inside any Neuralese stage (plans/neuralese/MAPLE_NESTED.md §4a, §9 item 5).

Owner goal (2026-10-08): every member ``LxE`` of a Maple-family student (Maple, Mellum) is a genuine objective in
every stage that moves shared weights, not only in ``nested_train``. Stages call these helpers so the family term and
its evaluation are one implementation:

- ``family_members``/``private_parameters``: the members of a loaded student and their private parts (router rows,
  norm gain corrections, private attention LoRA).
- ``member_backward``: one member's joint-phase objective on a token window, CE on the supervised targets plus
  KL(full ‖ member) to the current full model (detached teacher), exactly ``nested_train.member_step``'s terms.
- ``evaluate_members``: held CE and KL for the full model (KL to adapters-off original) and every member.
"""
from __future__ import annotations

import contextlib
import json
import random
from pathlib import Path

import torch

from ..train.joint_kd import IGNORE, chunked_ce_kl
from .nested_train import run
from .ternary import adapters_disabled


def family_members(backbone) -> list:
    """Members of the student under a port backbone (empty for a non-family backbone)."""
    return list(getattr(backbone.hf, "members", None) or [])


def private_parameters(backbone) -> list[tuple[str, torch.nn.Parameter]]:
    """Every member's private parts, named as in ``backbone.hf``."""
    return [(name, parameter) for name, parameter in backbone.hf.named_parameters()
            if ".private." in name or "private_gate." in name]


def window_labels(ids: torch.Tensor, prefix: int) -> torch.Tensor:
    """Next-token labels of a [B, T] window whose first ``prefix`` tokens are context only (IGNORE)."""
    labels = torch.full_like(ids, IGNORE)
    labels[:, prefix - 1:-1] = ids[:, prefix:]
    return labels


def leading_system_tokens(ids, start: list[int], end: int, bos: int | None = None) -> int:
    """Token count of a leading system message (through its end marker) when the window ``ids`` (a token list)
    begins with ``start`` (the tokens of e.g. "<|im_start|>system", after an optional BOS); 0 otherwise. Member terms
    keep it as context: a shared system prompt teaches members nothing but memorisation (MAPLE_NESTED §9.5)."""
    ids = list(ids)
    offset = 1 if bos is not None and ids[:1] == [bos] else 0
    if not start or ids[offset:offset + len(start)] != list(start):
        return 0
    try:
        return ids.index(end, offset + len(start)) + 1
    except ValueError:
        return len(ids)  # the window lies inside the system message


def member_backward(backbone, member, ids: torch.Tensor, labels: torch.Tensor, *, weight: float,
                    kl_weight: float = 1.0, chunk: int = 256, full_weight: float = 0.0):
    """Backpropagate ``weight · (CE_member + kl_weight · KL(full ‖ member))`` over the labelled positions; the full
    model is the detached teacher (the stage's own objective trains it). The member stays selected through backward,
    because checkpointed layers recompute their forward then and must route as the member did. ``full_weight`` adds
    the full model's own CE on the window (full_weight · CE_full), anchoring the shared weights the member term moves.
    Returns LossParts."""
    hf = backbone.hf
    head = hf.get_output_embeddings().weight
    if full_weight:
        full = run(hf, None, ids).last_hidden_state
        full_loss, _ = chunked_ce_kl(full, head, labels, chunk=chunk)
        (full_weight * full_loss).backward()
        full = full.detach()
    else:
        with torch.no_grad():
            full = run(hf, None, ids).last_hidden_state
    hf.set_member(member.key, experts=member.experts, layers=member.layers)
    try:
        hidden = hf.model(input_ids=ids).last_hidden_state
        loss, parts = chunked_ce_kl(hidden, head, labels, full, head, kl_weight=kl_weight, chunk=chunk)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite member loss ({member.key})")
        (weight * loss).backward()
    finally:
        hf.set_member(None)
    return parts


@torch.no_grad()
def evaluate_members(backbone, windows, *, chunk: int = 256) -> dict:
    """``windows``: iterable of (ids [1, T], labels [1, T]). Full model: CE and KL to the adapters-off original;
    members: CE and KL to the current full model. Token-weighted means."""
    head = backbone.hf.get_output_embeddings().weight
    totals: dict[str, list[float]] = {}

    def add(key, parts):
        row = totals.setdefault(key, [0.0, 0.0, 0])
        row[0] += parts.ce
        row[1] += parts.kl
        row[2] += parts.tokens

    for ids, labels in windows:
        with adapters_disabled():
            original = run(backbone.hf, None, ids).last_hidden_state
        full = run(backbone.hf, None, ids).last_hidden_state
        add("full", chunked_ce_kl(full, head, labels, original, head, kl_weight=1.0, chunk=chunk)[1])
        for member in family_members(backbone):
            hidden = run(backbone.hf, member, ids).last_hidden_state
            add(member.key, chunked_ce_kl(hidden, head, labels, full, head, kl_weight=1.0, chunk=chunk)[1])
    return {key: {"ce": c / max(n, 1), "kl": kl / max(n, 1), "tokens": n} for key, (c, kl, n) in totals.items()}


# The recipe component (owner 2026-10-12, REVIEW_20261012 rec. 4): the nested family as an in-stage objective -------------

SCHEMA = 'natlang.neuralese-family/1'


def parse_member(text: str) -> tuple[str, int, int | None]:
    """``"LxE"`` -> (key, layers, experts): the first L layers (an early exit through the final norm and the shared
    head) and the first E experts of the order per layer; ``"Lxall"``: every expert or a dense FFN (the nested axis
    of a dense backbone is depth)."""
    layers, _, experts = text.partition('x')
    if not layers.isdigit() or int(layers) < 1 or not (experts == 'all' or experts.isdigit() and int(experts) > 0):
        raise ValueError('a family member is "LxE" (first L layers, first E experts of the order, or "all"): ' + text)
    return f'{int(layers)}x{experts if experts == "all" else int(experts)}', int(layers), \
        None if experts == 'all' else int(experts)


def validate_family(spec, stages=None):
    """The recipe's ``family`` component (shared by both lines; the members and the order are backbone-inherent)::

        {"schema": "natlang.neuralese-family/1",
         "order": {"path": "runs/.../expert-order.pt", "sha256": "..."},   # N0 (maple/routing.py); MoE width only
         "members": {"28x32": 0.1, "21x32": 0.1},                            # trained members, loss weight each
         "evaluate": ["28x32", "28x24"],                                     # held CE columns (default: members)
         "kl_weight": 1.0, "seed": 0,
         "ramp": {"start": 0.03, "end": 0.08},                              # curriculum progress (the stage's)
         "stages": ["direct_sft", "direct_trajectory"]}

    Per update one objective is drawn (as the quantization component's ``sample`` mode): the full model with weight
    1, member m with ``weight_m · λ`` (λ the ramp mix at the stage's curriculum progress). A member update trains the
    member on the update's windows: the stage's own objective plus ``kl_weight`` x KL(full BF16 ‖ member) on the
    supervised positions, the full model (detached, BF16) the teacher. Members share every weight (no private parts),
    exit early after their L layers, route over the first E experts of the order, and train at the update's drawn
    precision point."""
    if not isinstance(spec, dict) or spec.get('schema') != SCHEMA:
        raise ValueError('family.schema must be ' + SCHEMA)
    unknown = set(spec) - {'schema', 'description', 'order', 'members', 'evaluate', 'kl_weight', 'seed', 'ramp',
                           'stages'}
    if unknown:
        raise ValueError('unknown family keys: ' + ', '.join(sorted(unknown)))
    order = spec.get('order')
    if order is not None and (not isinstance(order, dict) or set(order) != {'path', 'sha256'}):
        raise ValueError('family.order is {path, sha256} of an expert order (maple/routing.py)')
    members = spec.get('members')
    if not isinstance(members, dict) or not members:
        raise ValueError('family.members maps "LxE" to a positive loss weight')
    for key, weight in members.items():
        if parse_member(key)[0] != key or isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                or not weight > 0:
            raise ValueError('family.members maps "LxE" to a positive loss weight: ' + str(key))
    named = list(members) + list(spec.get('evaluate', []))
    for key in spec.get('evaluate', []):
        if parse_member(key)[0] != key:
            raise ValueError('family.evaluate lists "LxE" members: ' + str(key))
    if order is None and any(parse_member(key)[2] is not None for key in named):
        raise ValueError('expert-width members need family.order (the N0 expert order)')
    kl = spec.get('kl_weight', 1.0)
    if isinstance(kl, bool) or not isinstance(kl, (int, float)) or kl < 0:
        raise ValueError('family.kl_weight is a non-negative number')
    ramp = spec.get('ramp', {'start': 0.0, 'end': 0.0})
    if (not isinstance(ramp, dict) or set(ramp) != {'start', 'end'} or
            not 0 <= ramp['start'] <= ramp['end'] <= 1):
        raise ValueError('family.ramp is {start, end} in curriculum progress, 0 <= start <= end <= 1')
    named = spec.get('stages')
    if not isinstance(named, list) or not named or not all(isinstance(s, str) for s in named):
        raise ValueError('family.stages lists the stage ids that train the family')
    if stages is not None:
        missing = [s for s in named if s not in stages]
        if missing:
            raise ValueError('family names unknown stages: ' + ', '.join(missing))
        wrong = [s for s in named if stages[s] != 'neuralese_direct']
        if wrong:
            raise ValueError('the family objective is implemented for neuralese_direct stages: ' + ', '.join(wrong))


def stage_family(spec, stage_id):
    """The component as one stage receives it (``--family``), or None when the stage does not train the family."""
    if not spec or stage_id not in spec.get('stages', []):
        return None
    return {**json.loads(json.dumps(spec)), 'stage': stage_id}


class Family:
    """One stage's family objective over a port backbone (``backbone.forward_ids``; ``backbone.hf`` for experts)."""

    def __init__(self, component, *, root=None):
        validate_family({k: v for k, v in component.items() if k != 'stage'})
        self.component, self.stage = component, component.get('stage', '')
        self.weights = {key: float(w) for key, w in component['members'].items()}
        self.evaluated = list(component.get('evaluate') or self.weights)
        self.kl_weight = float(component.get('kl_weight', 1.0))
        self.seed = component.get('seed', 0)
        self.ramp = component.get('ramp', {'start': 0.0, 'end': 0.0})
        self.root = root
        self.members = {}

    def install(self, backbone):
        """Resolve the members against the backbone (and load, check and install the pinned expert order on its
        Maple-family model when a member restricts experts)."""
        layers = backbone.num_layers
        hf = backbone.hf
        mlp = getattr(getattr(hf.model, 'layers', [None])[0], 'mlp', None)
        experts = mlp.gate.weight.shape[0] if hasattr(mlp, 'gate') and hasattr(hf, 'set_member') else None
        installed = {}
        if self.component.get('order') is not None:
            from ..common.hashing import sha256_file_hex
            path = Path(self.component['order']['path'])
            if not path.is_absolute() and self.root is not None:
                path = Path(self.root) / path
            if sha256_file_hex(path) != self.component['order']['sha256']:
                raise ValueError('family.order changed since the recipe pinned it: ' + str(path))
            if experts is None:
                raise ValueError('family.order names an expert order, but the backbone has no experts')
            hf.set_member_order(torch.load(path, map_location='cpu')['orders'])
            installed['order'] = str(path)
        for key in dict.fromkeys(list(self.weights) + self.evaluated):
            _, depth, width = parse_member(key)
            if depth > layers or (width is not None and (experts is None or width >= experts)) or \
                    (depth == layers and width is None):
                raise ValueError(f'family member {key} is not a proper member of the model ({layers} layers, '
                                 f'{experts or "no"} experts)')
            self.members[key] = (depth, width)
        return {**installed, 'members': list(self.members), 'trained': self.weights}

    def mix(self, progress):
        from ..train.quantization import ramp_mix
        return ramp_mix(self.ramp, progress) if self.ramp['end'] > 0 else 1.0

    def sampled(self, step, progress):
        """The objective of update ``step``: None (the full model) or a trained member key, drawn by weight from
        ``Random((seed, stage, step))`` with the full model at weight 1 and member m at ``weight_m · λ``."""
        mix = self.mix(progress)
        candidates = [(None, 1.0)] + [(key, w * mix) for key, w in self.weights.items() if w * mix > 0]
        total = sum(w for _, w in candidates)
        draw = random.Random(f'{self.seed}:family:{self.stage}:{step}').random() * total
        for key, weight in candidates:
            draw -= weight
            if draw < 0:
                return key
        return candidates[-1][0]

    def describe(self, progress):
        return {'schema': SCHEMA, 'stage': self.stage, 'mix': self.mix(progress), 'trained': self.weights,
                'evaluated': self.evaluated, 'kl_weight': self.kl_weight}

    @contextlib.contextmanager
    def selected(self, backbone, key):
        """Run forwards (and their checkpointed recomputation in backward) as member ``key`` (None: full model)."""
        if key is None:
            yield
            return
        depth, width = self.members[key]
        backbone.exit_layers = depth if depth < backbone.num_layers else None
        if width is not None:
            backbone.hf.set_member(key, experts=width)
        try:
            yield
        finally:
            backbone.exit_layers = None
            if width is not None:
                backbone.hf.set_member(None)


def member_kl(backbone, hidden, teacher, labels, *, kl_weight, chunk=256):
    """``kl_weight`` x KL(teacher ‖ member) per labelled position over the backbone's readout (final norm, output
    matrix; the two control rows the serving readout patches are the output matrix's own here, for both sides), from
    pre-norm hidden states [B, S, W] (``teacher`` detached). Returns (loss, LossParts)."""
    weight = backbone.output_weight
    return chunked_ce_kl(backbone.final_norm(hidden), weight, labels, backbone.final_norm(teacher.detach()), weight,
                         ce_weight=0.0, kl_weight=kl_weight, chunk=chunk)
