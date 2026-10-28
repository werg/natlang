"""Own-output supervision objective (owner decision 2026-10-28).

The model's own outputs are the only CE targets: assistant replies/actions and
typed answers dominate, assistant reasoning gets a separately normalized share,
and prompts, instructions, inputs, tool/service output and harness feedback are
context with zero loss weight. Historical lineages keep their exact declared row
weights (``declared-row-weights/1``); this objective is selected only by an
explicit declaration in the stage's bound CE supervision policy.

Definition of ``own-outputs/1`` for one invocation (one graph node / one text
window), given per-token chat roles (``text_supervision.chat_roles``) and a
per-token eligibility mask:

1. A *decision* is a maximal run of assistant-role tokens (reasoning or reply).
2. Within a decision, eligible tokens are split into the reasoning class
   (``assistant_reasoning`` role, except the ``<think>``/``</think>`` markers,
   which belong to the reply/action class under ``reasoning_markers='reply'``:
   deciding when to stop reasoning and act is part of the action) and the
   reply/action class (everything else).
3. A decision with eligible reply tokens has mass ``1/D`` where ``D`` is the
   number of such decisions in the invocation; the reply class gets
   ``(1 - reasoning_share)`` of it and the reasoning class ``reasoning_share``,
   each spread uniformly over its tokens. Without eligible reasoning the reply
   class gets the whole decision. Reasoning without an eligible action (for
   example a truncated turn) is not a target.
4. Every other token (non-assistant roles, ineligible assistant tokens) has
   weight 0. Ineligible means: excluded by the row's own exact masks (failed
   historical actions, target holds, untrained reasoning, forced import
   positions, position 0), before an ordinary-text row's declared target
   suffix, or a runtime-opening assistant turn.
5. Each invocation with a decision has total mass 1, so every invocation in a
   graph counts equally; the native trainer normalizes each graph by its own
   mass and averages graphs, so a long transcript never outweighs many short
   decisions. An invocation without decisions has weight 0 everywhere (never
   an error).
6. Optional family balance multiplies each training graph's normalized loss
   by a bounded census-derived scale so each declared task family contributes
   equally in expectation (mean scale 1, capped at ``max_scale``).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from ..common.hashing import canonical_json_sha256_hex
from ..data.native_graph_schema import NativeGraphExample
from .text_supervision import ROLE_CODES, chat_roles

NAME = 'own-outputs/1'
LEGACY = 'declared-row-weights/1'
CATEGORIES = ('context', 'reply', 'reasoning', 'assistant_context')
CONTEXT, REPLY, REASONING, ASSISTANT_CONTEXT = range(len(CATEGORIES))
_ASSISTANT = frozenset((ROLE_CODES.index('assistant_reasoning'), ROLE_CODES.index('assistant_reply')))
_REASONING = ROLE_CODES.index('assistant_reasoning')
_FIXED = {'context_weight': 0., 'feedback_weight': 0., 'failed_assistant_weight': 0.,
          'balance': ['invocation', 'decision']}
_ROLE_NAMES = ('system', 'user', 'assistant', 'tool')
_TOOL_RESPONSE_PREFIXES = ('\n<tool_response>', '\n\n<tool_response>')
_SHA = re.compile('[0-9a-f]{64}')


@dataclass
class OwnOutputGraphExample(NativeGraphExample):
    """A training graph weighted by ``own-outputs/1``; categories are per flat position."""
    objective_categories: tuple = ()
    loss_scale: float = 1.
    objective_sha256: str = ''


def _token(value, *, optional=False):
    if value is None and optional:
        return None
    if type(value) is not int or value < 0:
        raise ValueError('own-output role tokens must be nonnegative integer IDs')
    return value


def _role_tokens(value):
    if not isinstance(value, dict) or set(value) != {'tokenizer_fingerprint', 'start', 'roles', 'think_open',
                                                      'think_close', 'tool_response_prefixes'}:
        raise ValueError('own-output objective needs exact role_tokens {tokenizer_fingerprint, start, roles, '
                         'think_open, think_close, tool_response_prefixes}')
    if not isinstance(value['tokenizer_fingerprint'], str) or not _SHA.fullmatch(value['tokenizer_fingerprint']):
        raise ValueError('own-output role tokens need the exact expanded tokenizer fingerprint')
    roles = value['roles']
    if not isinstance(roles, dict) or set(roles) != set(_ROLE_NAMES):
        raise ValueError('own-output role tokens name exactly system/user/assistant/tool')
    ids = [_token(roles[name]) for name in _ROLE_NAMES]
    start = _token(value['start'])
    think = (_token(value['think_open'], optional=True), _token(value['think_close'], optional=True))
    if len(set(ids)) != 4 or start in ids or (None in think) != (think[0] is None) or (
            think[0] is not None and (think[0] == think[1] or set(think) & {start, *ids})):
        raise ValueError('own-output role tokens must be distinct')
    prefixes = value['tool_response_prefixes']
    if not isinstance(prefixes, list) or any(not isinstance(p, list) or not p or any(
            type(t) is not int or t < 0 for t in p) for p in prefixes):
        raise ValueError('own-output tool-response prefixes are nonempty token-ID lists')
    return {'tokenizer_fingerprint': value['tokenizer_fingerprint'], 'start': start,
            'roles': {name: roles[name] for name in _ROLE_NAMES}, 'think_open': think[0], 'think_close': think[1],
            'tool_response_prefixes': [list(p) for p in prefixes]}


def _family_balance(value):
    if value is None:
        return None
    if not isinstance(value, dict) or value.get('key') not in ('source-cohort', 'source-group'):
        raise ValueError("family_balance is {key: 'source-cohort'|'source-group', [pattern], max_scale}")
    allowed = {'key', 'max_scale'} | ({'pattern'} if value['key'] == 'source-group' else set())
    scale = value.get('max_scale')
    if set(value) != allowed or type(scale) not in (int, float) or not math.isfinite(scale) or scale < 1:
        raise ValueError('family_balance needs exactly its key fields and a finite max_scale >= 1')
    result = {'key': value['key'], 'max_scale': float(scale)}
    if value['key'] == 'source-group':
        pattern = value['pattern']
        if not isinstance(pattern, str) or not pattern:
            raise ValueError('source-group family balance needs a nonempty regular expression')
        re.compile(pattern)
        result['pattern'] = pattern
    return result


def resolve(declaration):
    """Canonical objective declaration; ``None`` is the historical row-weight objective."""
    if declaration is None:
        return {'name': LEGACY}
    if not isinstance(declaration, dict):
        raise ValueError('supervision objective declaration must be an object')
    name = declaration.get('name')
    if name == LEGACY:
        if set(declaration) != {'name'}:
            raise ValueError('declared-row-weights/1 has no parameters')
        return {'name': LEGACY}
    if name != NAME:
        raise ValueError(f'unknown supervision objective {name!r}')
    allowed = {'name', 'reasoning_share', 'reasoning_markers', 'family_balance', 'role_tokens', *_FIXED}
    if set(declaration) - allowed:
        raise ValueError('unknown own-output objective fields: ' + ', '.join(sorted(set(declaration) - allowed)))
    for key, fixed in _FIXED.items():
        if key in declaration and declaration[key] != fixed:
            raise ValueError(f'own-outputs/1 fixes {key} = {fixed!r}')
    share = declaration.get('reasoning_share')
    if type(share) not in (int, float) or not math.isfinite(share) or not 0 <= share < 1:
        raise ValueError('own-output reasoning_share must be explicit, finite and in [0, 1)')
    markers = declaration.get('reasoning_markers', 'reply')
    if markers not in ('reply', 'reasoning'):
        raise ValueError("reasoning_markers is 'reply' or 'reasoning'")
    return {'name': NAME, 'reasoning_share': float(share), 'reasoning_markers': markers, **_FIXED,
            'family_balance': _family_balance(declaration.get('family_balance')),
            'role_tokens': _role_tokens(declaration.get('role_tokens'))}


def identity(objective):
    """Exact identity of a resolved objective (what checkpoints record and continuations compare)."""
    return {'name': objective['name'], 'sha256': canonical_json_sha256_hex(objective)}


def policy_objective(policy):
    """Resolved objective of a loaded CE policy (``None`` policy or no ``objective`` key: legacy)."""
    return resolve(None if policy is None else policy.get('objective'))


def role_tokens_from_tokenizer(tokenizer, *, tokenizer_fingerprint):
    """Exact chat-role token IDs of a ChatML tokenizer (LFM2.5, LFM3, Mellum); never guessed."""
    def exact(text):
        token = tokenizer.convert_tokens_to_ids(text)
        if type(token) is not int or token == getattr(tokenizer, 'unk_token_id', None):
            return None
        return token
    start = exact('<|im_start|>')
    roles = {name: exact(name) for name in _ROLE_NAMES}
    if start is None or None in roles.values():
        raise ValueError('own-output role labelling needs single-token <|im_start|> and role names')
    for name, token in roles.items():
        rendered = tokenizer('<|im_start|>' + name + '\n', add_special_tokens=False)['input_ids']
        if list(rendered[:2]) != [start, token]:
            raise ValueError(f'rendered <|im_start|>{name} does not start with its exact role token')
    prefixes = [list(tokenizer(prefix, add_special_tokens=False)['input_ids']) for prefix in _TOOL_RESPONSE_PREFIXES]
    return _role_tokens({'tokenizer_fingerprint': tokenizer_fingerprint, 'start': start, 'roles': roles,
                         'think_open': exact('<think>'), 'think_close': exact('</think>'),
                         'tool_response_prefixes': prefixes})


def label_roles(ids, role_tokens):
    """``text_supervision.chat_roles`` with the objective's exact declared role tokens."""
    return chat_roles(ids, start_id=role_tokens['start'],
                      role_ids={token: name for name, token in role_tokens['roles'].items()},
                      think_open=role_tokens['think_open'], think_close=role_tokens['think_close'],
                      tool_response_prefixes=tuple(tuple(p) for p in role_tokens['tool_response_prefixes']))


def assistant_runs(labels):
    """Maximal assistant-role runs ``[begin, end)``: the invocation's decisions."""
    runs, index = [], 0
    while index < len(labels):
        if labels[index] not in _ASSISTANT:
            index += 1
            continue
        begin = index
        while index < len(labels) and labels[index] in _ASSISTANT:
            index += 1
        runs.append((begin, index))
    return runs


def sequence_weights(labels, ids, eligible, *, reasoning_share, reasoning_markers='reply', think_ids=()):
    """Weights and categories of one invocation under ``own-outputs/1`` (module docstring, steps 1-5).

    Returns ``(weights, categories)``; weights sum to 1 when the invocation has a decision, else 0."""
    size = len(labels)
    if len(ids) != size or len(eligible) != size:
        raise ValueError('own-output roles, token IDs and eligibility must align')
    if not 0 <= reasoning_share < 1:
        raise ValueError('reasoning share must lie in [0, 1)')
    markers = set(think_ids) if reasoning_markers == 'reply' else set()
    categories = [CONTEXT] * size
    decisions = []
    for begin, end in assistant_runs(labels):
        reply, reasoning = [], []
        for position in range(begin, end):
            categories[position] = ASSISTANT_CONTEXT
            if not eligible[position]:
                continue
            if labels[position] == _REASONING and ids[position] not in markers:
                reasoning.append(position)
            else:
                reply.append(position)
        if reply:
            decisions.append((reply, reasoning))
    weights = [0.] * size
    for reply, reasoning in decisions:
        mass = 1. / len(decisions)
        share = reasoning_share if reasoning else 0.
        for position in reply:
            weights[position] = mass * (1. - share) / len(reply)
            categories[position] = REPLY
        for position in reasoning:
            weights[position] = mass * share / len(reasoning)
            categories[position] = REASONING
    return weights, categories


def window_weights(roles, *, reasoning_share, ids=None, think_ids=(), reasoning_markers='reply',
                   suffix_starts=None, spans=None, exclude=None):
    """``own-outputs/1`` for prepared text windows ([B, S] role codes), the counterpart of
    ``text_supervision.balanced_position_weights``/``segmented_position_weights``.

    Positions before a window's ``suffix_starts`` entry, outside its ``spans`` (a list of ``[begin, end)`` lists
    per window) or in ``exclude`` are context. Each window with a decision sums to 1; a window without any
    eligible assistant target is all zero (never an error). Returns ``(weights [B, S] float32, categories)``."""
    import torch
    batch, size = roles.shape
    if ids is not None and ids.shape != roles.shape or exclude is not None and exclude.shape != roles.shape:
        raise ValueError('token IDs and exclusions must align with role labels')
    if suffix_starts is not None and len(suffix_starts) != batch or spans is not None and len(spans) != batch:
        raise ValueError('one suffix coordinate or span list per window required')
    result, categories = [], []
    for row in range(batch):
        eligible = [True] * size
        if suffix_starts is not None and suffix_starts[row] is not None:
            eligible = [position >= suffix_starts[row] for position in range(size)]
        if spans is not None:
            inside = [False] * size
            for begin, end in spans[row]:
                if not 0 <= begin <= end <= size:
                    raise ValueError(f'target span outside its window: {(begin, end)}')
                inside[begin:end] = [True] * (end - begin)
            eligible = [a and b for a, b in zip(eligible, inside)]
        if exclude is not None:
            eligible = [a and not bool(b) for a, b in zip(eligible, exclude[row].tolist())]
        weights, category = sequence_weights(roles[row].tolist(), ids[row].tolist() if ids is not None
                                             else [-1] * size, eligible, reasoning_share=reasoning_share,
                                             reasoning_markers=reasoning_markers, think_ids=think_ids)
        result.append(weights)
        categories.append(category)
    return torch.tensor(result, dtype=torch.float32, device=roles.device).reshape(batch, size), categories


def _think_ids(role_tokens):
    return tuple(token for token in (role_tokens['think_open'], role_tokens['think_close']) if token is not None)


def _node_suffixes(row, raw_nodes):
    conversion = row.get('conversion') or {}
    if conversion.get('role_interpretation') == 'ordinary-text' and 'source_suffix_start' in conversion:
        suffix = conversion['source_suffix_start']
        if len(raw_nodes) != 1 or type(suffix) is not int or not 0 <= suffix <= len(raw_nodes[0]):
            raise ValueError('own-output ordinary row needs one node and an exact suffix inside it')
        return [suffix]
    if row.get('source_pairing'):
        # The witnessed pair encoding's exact target suffix: its first full-weight token.
        return [next((i for i, value in enumerate(raw) if value == 1.), len(raw)) for raw in raw_nodes]
    return [0] * len(raw_nodes)


def graph_weights(row, objective):
    """Own-output weights and categories for a complete authenticated native graph row (all nodes, flat)."""
    role_tokens = objective['role_tokens']
    if row.get('tokenizer_fingerprint') != role_tokens['tokenizer_fingerprint']:
        raise ValueError('own-output role tokens belong to a different tokenizer than this graph')
    conversion = row.get('conversion') or {}
    nodes = row['nodes']
    raw_nodes = [node['target_weights'] for node in nodes]
    suffixes = _node_suffixes(row, raw_nodes)
    ordinary = conversion.get('role_interpretation') == 'ordinary-text' and 'source_suffix_start' in conversion
    opening = (conversion.get('runtime_opening_assistant_turns', 0)
               if conversion.get('role_interpretation') == 'whole-trajectory-chat-roles' else 0)
    if type(opening) is not int or opening < 0:
        raise ValueError('runtime-opening assistant turns must be a nonnegative integer')
    think = _think_ids(role_tokens)
    weights, categories = [], []
    for index, (node, raw, suffix) in enumerate(zip(nodes, raw_nodes, suffixes)):
        ids = node['token_ids']
        labels = list(label_roles(ids, role_tokens))
        if ordinary:
            # The declared suffix is the exact assistant target. Literal chat-boundary text inside it must not
            # silently shrink the target, so the suffix keeps assistant roles (reasoning where parsed).
            labels[suffix:] = [code if code in _ASSISTANT else ROLE_CODES.index('assistant_reply')
                               for code in labels[suffix:]]
        eligible = [value > 0 and position >= suffix for position, value in enumerate(raw)]
        if index == 0 and opening:
            runs = assistant_runs(labels)
            if opening > len(runs):
                raise ValueError('runtime-opening assistant turns exceed the source trajectory')
            for begin, end in runs[:opening]:
                eligible[begin:end] = [False] * (end - begin)
        node_weights, node_categories = sequence_weights(labels, ids, eligible,
            reasoning_share=objective['reasoning_share'], reasoning_markers=objective['reasoning_markers'],
            think_ids=think)
        weights.extend(node_weights)
        categories.extend(node_categories)
    return weights, categories


def apply_to_example(example, row, objective, *, family_scales=None):
    """Replace a TRAIN example's weights by the declared objective (no row identity or credit changes)."""
    if example.split != 'train':
        raise ValueError('own-output training weights apply to TRAIN rows; held rows get measurements')
    digest = identity(objective)['sha256']
    declared = (row.get('conversion') or {}).get('supervision_policy') or {}
    if declared.get('name') == NAME:
        if declared.get('objective_sha256') != digest:
            raise ValueError('row was converted under a different own-output objective')
    weights, categories = graph_weights(row, objective)
    if declared.get('name') == NAME and any(abs(a - b) > 1e-12 * max(1., abs(b))
                                            for a, b in zip(weights, example.target_weights)):
        raise ValueError('converted own-output row weights differ from the declared objective')
    if len(weights) != len(example.target_weights) or any(
            w > 0 and raw <= 0 for w, raw in zip(weights, example.target_weights)):
        raise ValueError('own-output weights must stay inside the row\'s exact positive positions')
    scale = 1.
    if objective['family_balance'] is not None:
        if family_scales is None or example.id not in family_scales:
            raise ValueError('family-balanced own-output objective requires its TRAIN census')
        scale = family_scales[example.id]
    from dataclasses import fields
    return OwnOutputGraphExample(**{item.name: getattr(example, item.name) for item in fields(NativeGraphExample)
                                    if item.name != 'target_weights'},
                                 target_weights=tuple(weights), objective_categories=tuple(categories),
                                 loss_scale=scale, objective_sha256=digest)


def held_measurement_weights(row, objective):
    """Evaluation-only sidecars on held rows: the objective and per-category CE (context is monitoring only)."""
    weights, categories = graph_weights(row, objective)
    def only(code):
        return tuple(1. if category == code else 0. for category in categories)
    return {'own_outputs': tuple(weights), 'own_reply': only(REPLY), 'own_reasoning': only(REASONING),
            'context_monitor': only(CONTEXT), 'assistant_context_monitor': only(ASSISTANT_CONTEXT)}


def attach_measurements(example, row, objective, *, source_sha256):
    from .native_supervision import NativeHeldGraphExample, attach_held_measurements
    sidecars = held_measurement_weights(row, objective)
    if len(sidecars['own_outputs']) != len(example.target_weights):
        raise ValueError('held own-output measurements do not align with the graph')
    binding = {'source_sha256': source_sha256, 'objective': identity(objective)}
    if isinstance(example, NativeHeldGraphExample):
        if set(sidecars) & set(example.evaluation_weights):
            raise ValueError('held measurement names collide')
        from dataclasses import replace
        return replace(example, evaluation_weights={**example.evaluation_weights, **sidecars},
                       evaluation_binding={**example.evaluation_binding, 'objective': binding['objective']})
    return attach_held_measurements(example, sidecars, binding)


def family_scales(objective, entries, *, cohort_ids=None):
    """Bounded per-graph scales that equalize declared families in expectation, with a census receipt."""
    balance = objective['family_balance']
    if balance is None:
        return None, None
    if balance['key'] == 'source-cohort':
        if cohort_ids is None or len(cohort_ids) != len(entries):
            raise ValueError('source-cohort family balance needs one cohort identity per TRAIN row')
        families = list(cohort_ids)
    else:
        pattern = re.compile(balance['pattern'])
        families = []
        for entry in entries:
            family = '<unmatched>'
            for group in entry['groups']:
                match = pattern.search(group)
                if match:
                    family = match.group(1) if match.groups() else match.group(0)
                    break
            families.append(family)
    counts = {}
    for family in families:
        counts[family] = counts.get(family, 0) + 1
    total, number = len(families), len(counts)
    capped = {family: min(balance['max_scale'], total / (number * count)) for family, count in counts.items()}
    normal = total / sum(capped[family] * count for family, count in counts.items())
    per_family = {family: capped[family] * normal for family in counts}
    scales = {entry['id']: per_family[family] for entry, family in zip(entries, families)}
    census = {'families': {family: {'rows': counts[family], 'scale': per_family[family]} for family in sorted(counts)},
              'rows': total, 'key': balance['key']}
    census['sha256'] = canonical_json_sha256_hex(census)
    return scales, census


def prepare_policy(policy, train):
    """Attach the TRAIN family census to a loaded policy (private keys; the bound file is unchanged)."""
    if policy is None:
        return None
    objective = policy_objective(policy)
    if objective['name'] != NAME or objective['family_balance'] is None:
        return policy
    cohort_ids = None
    if objective['family_balance']['key'] == 'source-cohort':
        from .native_cohort_schedule import source_cohort_ids
        cohort_ids = source_cohort_ids(train)
    scales, census = family_scales(objective, train.entries, cohort_ids=cohort_ids)
    return {**policy, '_family_scales': scales, '_family_census': census}


def continuation_identity(previous_options, policy):
    """The objective identity a run records, refusing a silent objective change on continuation.

    ``previous_options`` are the continued checkpoint's options (``None``/empty: a fresh run). A change requires the
    new policy's explicit ``objective_transition: {from: <previous sha256>}``; it is recorded, never inferred."""
    current = identity(policy_objective(policy))
    census = (policy or {}).get('_family_census')
    if census is not None:
        current['family_census_sha256'] = census['sha256']
    if not previous_options:
        return current, None
    previous = previous_options.get('supervision_objective') or identity(resolve(None))
    if previous['sha256'] == current['sha256']:
        return current, None
    transition = (policy or {}).get('objective_transition')
    if not isinstance(transition, dict) or transition.get('from') != previous['sha256']:
        raise ValueError(f"continuation changes the supervision objective {previous['name']} -> {current['name']} "
                         'without an explicit objective_transition from the previous objective sha256')
    return current, {'from': {k: previous[k] for k in ('name', 'sha256')},
                     'to': {k: current[k] for k in ('name', 'sha256')}}


def validate_policy_fields(value):
    """CE-policy schema checks for the objective keys (called by ``native_supervision.load_policy``)."""
    if 'objective' in value:
        resolve(value['objective'])
    transition = value.get('objective_transition')
    if transition is not None and (not isinstance(transition, dict) or set(transition) != {'from'}
                                   or not isinstance(transition['from'], str) or not _SHA.fullmatch(transition['from'])):
        raise ValueError('objective_transition is {from: <previous objective sha256>}')


def progress_telemetry(details, *, graphs):
    """Per-category TRAIN CE of one update (decision-weighted means; context CE is measured on held rows)."""
    measured = [item['own_output'] for item in details if item.get('own_output')]
    result = {'objective': NAME, 'graphs': graphs, 'targeted_graphs': len(details),
              'measured_graphs': len(measured),
              'mean_loss_scale': sum(item.get('loss_scale', 1.) for item in details) / max(len(details), 1)}
    for name in ('reply', 'reasoning'):
        weight = sum(item[name]['weight'] for item in measured)
        loss = sum(item[name]['loss_sum'] for item in measured)
        result[name] = {'weight': weight, 'loss': loss / weight if weight > 0 else None}
    return result


def conversion_weights(ids, eligible, objective, *, tokenizer, tokenizer_fingerprint, suffix=None):
    """Conversion-time ``own-outputs/1`` weights for one rendered invocation, with its row declaration.

    ``eligible`` already excludes failed/held/runtime-opening turns; ``suffix`` (ordinary rows) marks the exact
    assistant target. The declared role tokens must equal the converting tokenizer's exact chat-role tokens."""
    objective = resolve(objective)
    if objective['name'] != NAME:
        raise ValueError('conversion-time objective must be own-outputs/1')
    if role_tokens_from_tokenizer(tokenizer, tokenizer_fingerprint=tokenizer_fingerprint) != objective['role_tokens']:
        raise ValueError('declared own-output role tokens differ from the converting tokenizer')
    labels = list(label_roles(ids, objective['role_tokens']))
    if suffix is not None:
        labels[suffix:] = [code if code in _ASSISTANT else ROLE_CODES.index('assistant_reply')
                           for code in labels[suffix:]]
        eligible = [value and position >= suffix for position, value in enumerate(eligible)]
    eligible = [bool(value) and position > 0 for position, value in enumerate(eligible)]
    weights, categories = sequence_weights(labels, ids, eligible, reasoning_share=objective['reasoning_share'],
        reasoning_markers=objective['reasoning_markers'], think_ids=_think_ids(objective['role_tokens']))
    declaration = {'name': NAME, 'objective_sha256': identity(objective)['sha256'],
                   'reasoning_share': objective['reasoning_share'],
                   'reasoning_markers': objective['reasoning_markers'],
                   'context_weight': 0., 'feedback_weight': 0., 'failed_assistant_weight': 0.,
                   'decisions': sum(1 for begin, end in assistant_runs(labels)
                                    if any(categories[i] == REPLY for i in range(begin, end))),
                   'reply_tokens': categories.count(REPLY), 'reasoning_tokens': categories.count(REASONING),
                   'role_parser': 'natlang_neuralese.train.text_supervision.chat_roles',
                   'weighting': 'natlang_neuralese.train.own_output_supervision.sequence_weights'}
    return weights, declaration
