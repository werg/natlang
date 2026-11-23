"""Finite decision readouts (natlang ``decide``) as native graph supervision.

A decision readout is one model invocation: the runtime renders the call's decision prompt once and scores every
allowed reply as the complete closed assistant continuation (end of message included), exactly like the serving
``native_learning._score_rendered``. Its output is the choice over that finite option set, so it is the model's own
output even though no token is sampled.

In a native graph row a decision is K sibling *option nodes*. Option node i holds the identical prompt (the same
token IDs and the same native-state imports at the same positions) followed by option i's continuation tokens.
Option nodes carry zero token target weights; the decision objective supervises the choice over the option set
(``train.decision_supervision``). Row field::

    "decisions": [{"name": str, "options": [option node names, in option order], "prompt_tokens": int,
                   "label": int | "probabilities": [p_0..p_{K-1}], "label_source": "exact-oracle" |
                   "verified-trajectory", "values": [reply texts]}]

``exact-oracle`` labels come from an exact oracle of the decision itself; ``verified-trajectory`` labels are the
recorded choice of a trajectory whose root outcome was verified. Sibling option nodes share causal prefixes, so the
prefix packer folds their prompts physically where the trainer enables it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

LABEL_SOURCES = ('exact-oracle', 'verified-trajectory')
# The runtime's decision readout prompt (ts-host/src/native/decision.ts), byte for byte; parity:
# tests/neuralese/test_decision_supervision.py::test_decision_prompt_parity_with_the_typescript_runtime.
DECISION_SYSTEM_PROMPT = ('You are running one call of a natural-language function. Carry out its '
                          "instructions yourself for this call's argument values, judging the visible inputs. Your reply is "
                          "the function's return value: one JSON value of the declared type, with nothing else.")


DECISION_REPLY_REQUEST = 'Reply with exactly one of these JSON values and nothing else: '


def decision_prompt(replies):
    return 'Answer this call now, without tools. ' + DECISION_REPLY_REQUEST + ', '.join(replies) + '.'


def decision_replies(messages):
    """The reply texts a readout request offers (the runtime's ``decisionPrompt``, ts-host/src/decision-frame.ts, and
    the refinement judge's question end the last user message with ``DECISION_REPLY_REQUEST`` + the JSON texts joined by
    ', ' + '.'), in option order; None when the last message is not such a request or offers fewer than two distinct
    values. The texts are the scorer's options byte for byte: each is the JSON of one value (``JSON.stringify``)."""
    import json
    last = messages[-1] if messages else None
    content = last.get('content') if isinstance(last, dict) and last.get('role') == 'user' else None
    if not isinstance(content, str) or DECISION_REPLY_REQUEST not in content or not content.endswith('.'):
        return None
    text = content.rsplit(DECISION_REPLY_REQUEST, 1)[1][:-1]
    decoder, replies, at = json.JSONDecoder(), [], 0
    while True:
        try:
            _, end = decoder.raw_decode(text, at)
        except ValueError:
            return None
        replies.append(text[at:end])
        if end == len(text):
            break
        if not text.startswith(', ', end):
            return None
        at = end + 2
    return replies if len(replies) >= 2 and len(set(replies)) == len(replies) else None


DECISION_FORM_REQUEST = 'Reply with exactly one JSON value of this form and nothing else: '
DECISION_FORMS_REQUEST = 'Reply with exactly one JSON value of one of these forms and nothing else: '


def _form(text, at, decoder):
    """One FORM of a field-mode request at ``at``: ('value', JSON text) for a scalar arm, ('record', [(key JSON text,
    [alternative JSON texts])]) for a record arm whose field values are ``<ALT | ALT | ...>`` placeholders (a bare JSON
    value is a field with one alternative); returns (form, end)."""
    if not text.startswith('{', at) or text.startswith('{}', at):   # a scalar arm, or a record with no fields
        _, end = decoder.raw_decode(text, at)
        return ('value', text[at:end]), end
    fields, at = [], at + 1
    while True:
        key, end = decoder.raw_decode(text, at)
        if not isinstance(key, str) or not text.startswith(':', end):
            raise ValueError('a form field is "key":value')
        key_text, at = text[at:end], end + 1
        if text.startswith('<', at):
            alternatives, at = [], at + 1
            while True:
                _, end = decoder.raw_decode(text, at)
                alternatives.append(text[at:end])
                if text.startswith(' | ', end):
                    at = end + 3
                elif text.startswith('>', end):
                    at = end + 1
                    break
                else:
                    raise ValueError('placeholder alternatives are separated by " | " and closed by ">"')
        else:
            _, end = decoder.raw_decode(text, at)
            alternatives, at = [text[at:end]], end
        if not alternatives or len(set(alternatives)) != len(alternatives):
            raise ValueError('a field needs distinct alternatives')
        fields.append((key_text, alternatives))
        if text.startswith(',', at):
            at += 1
        elif text.startswith('}', at):
            return ('record', fields), at + 1
        else:
            raise ValueError('form fields are separated by "," and closed by "}"')


def decision_forms(messages):
    """The arms a field-mode readout request offers (the runtime's ``decisionFieldsPrompt``: the last user message ends
    with ``DECISION_FORM_REQUEST`` or ``DECISION_FORMS_REQUEST`` + FORM texts joined by ', ' + '.'), in order (``_form``
    shapes); None when the last message is not such a request. Every JSON text is read with raw_decode."""
    import json
    last = messages[-1] if messages else None
    content = last.get('content') if isinstance(last, dict) and last.get('role') == 'user' else None
    if not isinstance(content, str) or not content.endswith('.'):
        return None
    marker = next((m for m in (DECISION_FORMS_REQUEST, DECISION_FORM_REQUEST) if m in content), None)
    if marker is None:
        return None
    text = content.rsplit(marker, 1)[1][:-1]
    decoder, forms, at = json.JSONDecoder(), [], 0
    while True:
        try:
            form, at = _form(text, at, decoder)
        except (ValueError, IndexError):
            return None
        forms.append(form)
        if at == len(text):
            break
        if not text.startswith(', ', at):
            return None
        at += 2
    if (marker == DECISION_FORM_REQUEST) != (len(forms) == 1):
        return None
    return forms


def _arm_candidate(form, values):
    """Arm ``form``'s open reply text through the chosen field values ``values`` (its whole text when complete)."""
    if form[0] == 'value':
        return form[1]
    fields = form[1]
    body = ','.join(f'{key}:{value}' for (key, _), value in zip(fields, values))
    return '{' + body + ('}' if len(values) == len(fields) else ',"')


def field_candidates(forms, chosen):
    """The candidates of the next field-mode step after the chosen candidate texts ``chosen`` (a list, step order):
    [(candidate text, complete)] in arm and alternative order, distinct; [] when ``chosen`` is complete. The arms
    still in play are those whose text matched every chosen candidate; each step extends their chosen values by one
    field (a scalar arm is one step)."""
    live = [(form, []) for form in forms]
    for text in chosen:
        nxt = []
        for form, values in live:
            if form[0] == 'value':
                if not values and form[1] == text:
                    nxt.append((form, [form[1]]))
                continue
            if len(values) < len(form[1]):
                for alternative in form[1][len(values)][1]:
                    if _arm_candidate(form, values + [alternative]) == text:
                        nxt.append((form, values + [alternative]))
        live = nxt
    out = []
    for form, values in live:
        if form[0] == 'value':
            if not values:
                out.append((form[1], True))
        elif len(values) < len(form[1]):
            for alternative in form[1][len(values)][1]:
                out.append((_arm_candidate(form, values + [alternative]), len(values) + 1 == len(form[1])))
    seen, distinct = set(), []
    for candidate in out:
        if candidate[0] not in seen:
            seen.add(candidate[0])
            distinct.append(candidate)
    return distinct


def field_path(forms, reply):
    """The field-mode steps a complete reply text takes: [(candidates, label)] per step, ``candidates`` the step's
    candidate texts and ``label`` the reply's (teacher forcing of the reply's own choices: an arm whose fields, in
    order, each equal one of its alternatives as a JSON value, else whose scalar equals the reply); None when the reply
    answers no arm. A reply in another key order or spacing is matched by value; the steps are the canonical texts."""
    import json
    try:
        value = json.loads(reply)
    except (TypeError, ValueError):
        return None
    same = lambda a, b: a == b and type(a) is type(b)
    path = None
    for form in forms:
        if form[0] == 'value':
            if same(json.loads(form[1]), value):
                path = [form[1]]
        elif isinstance(value, dict) and sorted(value) == sorted(json.loads(key) for key, _ in form[1]):
            values = []
            for key, alternatives in form[1]:
                match = [a for a in alternatives if same(json.loads(a), value[json.loads(key)])]
                if len(match) != 1:
                    break
                values.append(match[0])
            else:
                path = [_arm_candidate(form, values[:index + 1]) for index in range(len(values))]
        if path is not None:
            break
    if path is None:
        return None
    steps = []
    for index, label in enumerate(path):
        candidates = [text for text, _ in field_candidates(forms, path[:index])]
        if label not in candidates:
            return None
        steps.append((candidates, candidates.index(label)))
    return steps


def decision_label(replies, reply):
    """The index of the option ``reply`` (an assistant reply text) answers: its exact text, else the option whose JSON
    value it spells (surrounding whitespace, other spacing); None when it answers none of them."""
    import json
    if not isinstance(reply, str):
        return None
    if reply in replies:
        return replies.index(reply)
    try:
        value = json.loads(reply)
    except ValueError:
        return None
    matches = [index for index, option in enumerate(replies) if json.loads(option) == value
               and type(json.loads(option)) is type(value)]
    return matches[0] if len(matches) == 1 else None


def decision_messages(opening, values, *, guidance=''):
    """The readout's messages for a finite-typed call (agent.ts decisionReadout): the decision system prompt (plus
    program guidance), the call's opening without its system turn and the answer request over the JSON values."""
    import json
    replies = [json.dumps(value, ensure_ascii=False, separators=(',', ':')) for value in values]
    return ([{'role': 'system', 'content': DECISION_SYSTEM_PROMPT + guidance}, *opening[1:],
             {'role': 'user', 'content': decision_prompt(replies)}], replies)


# ---- reasoned readouts (DECISIONS 2026-11-23 "reasoned decisions") --------------------------------------------------
# A reasoned readout samples k reasonings R_j (generation turns, trajectory phase ``decision_reasoning``) and scores the
# options after each (``serve.native_learning.decision_items(..., reasoning=R_j)``): k decision entries of phase
# ``action`` with ``decision_reasoning: {index, samples, action}``; the action entry (``action: true``) is the one whose
# sampled choice the program used. ``rl.task_rollouts.rollout_turns`` keeps the entries as turns (their reasoning
# generations never become turns) and adds the paired generation's sampled ids (``reasoning_sample_ids``) and the
# scored prompt's hash (``scored_prompt_sha256``).

def token_ids_sha256(ids):
    """The hash a reasoned decision turn records of its scored prompt ids (a capture's token prompt)."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps([int(i) for i in ids], separators=(',', ':')).encode()).hexdigest()


def capture_prompt_ids(capture):
    """A capture's prompt as token ids, or None when it holds anything but tokens (imported native state)."""
    prompt = (capture or {}).get('prompt') if isinstance(capture, dict) else None
    if not isinstance(prompt, list) or not prompt or any(
            not isinstance(part, dict) or part.get('kind') != 'tokens' for part in prompt):
        return None
    return [int(i) for part in prompt for i in part.get('token_ids') or ()]


def reasoned_decision_plan(turns):
    """Per turn of an episode (``natlang.rollout/1`` semantic turns): None for a turn that is no reasoned decision
    entry, else {'group': n, 'train': bool, 'why': 'agrees' | 'answer-differs' | 'no-action' | 'several-actions'}.

    The training rule (spec "Training rule"): of a reasoned readout's samples, those whose reply equals the action
    entry's reply (the answer the program used) train; the others do not; a readout without exactly one action entry
    trains none. A readout's entries share the invocation and the messages; one sample index appears once per readout,
    except in a field-mode readout, where sample j's later steps extend its earlier step's reply (each step is an entry,
    the sample's reply is its last)."""
    import json
    groups, open_groups, plan = [], {}, [None] * len(turns)
    for at, turn in enumerate(turns):
        dr = turn.get('decision_reasoning') if isinstance(turn, dict) else None
        if not isinstance(dr, dict):
            continue
        key = (turn.get('invocation_id'), json.dumps(turn.get('messages'), sort_keys=True, ensure_ascii=False))
        reply = ((turn.get('target') or {}).get('content') or '').strip()
        index = dr.get('index')
        group = open_groups.get(key)
        if group is not None and index in group['replies']:
            earlier = group['replies'][index]
            if not (reply.startswith(earlier) and reply != earlier):   # not a later field step: a new readout
                group = None
        if group is None or dr.get('samples') != group['samples']:
            group = open_groups[key] = {'members': [], 'replies': {}, 'samples': dr.get('samples'), 'actions': set()}
            groups.append(group)
        group['members'].append((at, index))
        group['replies'][index] = reply
        if dr.get('action') is True:
            group['actions'].add(index)
    for number, group in enumerate(groups):
        actions = group['actions']
        chosen = group['replies'][next(iter(actions))] if len(actions) == 1 else None
        for at, index in group['members']:
            if chosen is None:
                why = 'no-action' if not actions else 'several-actions'
            else:
                why = 'agrees' if group['replies'][index] == chosen else 'answer-differs'
            plan[at] = {'group': number, 'train': why == 'agrees', 'why': why}
    return plan


def reasoning_span(tokenizer, at, prompt_ids):
    """(start, stop) of a reasoned readout's reasoning in its decision prompt ids: from ``at`` (the end of the
    generation prompt, after ``<think>``) through the first think close (``</think>``) included, the student's own
    sampled tokens. Raises when the prompt has no think close after ``at``."""
    close = tokenizer.convert_tokens_to_ids('</think>')
    ids = [int(i) for i in prompt_ids]
    if close is None or close == getattr(tokenizer, 'unk_token_id', None) or close not in ids[at:]:
        raise ValueError('a reasoned decision prompt closes its reasoning with </think>')
    return at, ids.index(close, at) + 1


_FIELDS = {'name', 'options', 'prompt_tokens', 'label', 'probabilities', 'label_source', 'values'}


@dataclass(frozen=True)
class GraphDecision:
    """One parsed decision of a graph: option node names, the shared prompt length and the target distribution."""
    name: str
    options: tuple
    prompt_tokens: int
    target: tuple
    label_source: str


def option_node_name(name, index):
    return f'{name}:option:{index}'


def parse_decisions(row, nodes, weights):
    """Validate a row's ``decisions`` against its parsed nodes and flat target weights (no decisions: ``()``)."""
    declared = row.get('decisions')
    if declared is None:
        return ()
    if not isinstance(declared, list) or not declared:
        raise ValueError('native graph decisions must be a nonempty list when present')
    by_name, offsets, offset = {}, {}, 0
    for node in nodes:
        by_name[node.name] = node
        offsets[node.name] = offset
        offset += len(node.token_ids)
    imported = {edge.source for node in nodes for edge in node.imports}
    used, names, result = set(), set(), []
    for value in declared:
        if not isinstance(value, dict) or set(value) - _FIELDS or not {'name', 'options', 'prompt_tokens',
                                                                       'label_source'} <= set(value):
            raise ValueError('native decision needs exactly name, options, prompt_tokens, label or probabilities, '
                             'label_source and optional values')
        name, options, prompt = value['name'], value['options'], value['prompt_tokens']
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('native decision names must be unique nonempty strings')
        names.add(name)
        if (not isinstance(options, list) or len(options) < 2 or len(set(options)) != len(options)
                or any(option not in by_name or option in used for option in options)):
            raise ValueError('native decision needs at least two distinct, unshared option nodes of its graph')
        used.update(options)
        if value['label_source'] not in LABEL_SOURCES:
            raise ValueError(f'native decision label_source must be one of {LABEL_SOURCES}')
        first = by_name[options[0]]
        if type(prompt) is not int or prompt < 1:
            raise ValueError('native decision prompt_tokens must be a positive integer')
        continuations = set()
        for option in options:
            node = by_name[option]
            if (len(node.token_ids) <= prompt or node.token_ids[:prompt] != first.token_ids[:prompt]
                    or node.imports != first.imports or any(edge.at >= prompt for edge in node.imports)):
                raise ValueError('native decision options must share one exact prompt and its imports, '
                                 'each followed by its own nonempty continuation')
            if node.producing or node.logical_producing or option in imported:
                raise ValueError('native decision options neither produce nor export state')
            start = offsets[option]
            if any(weight != 0 for weight in weights[start:start + len(node.token_ids)]):
                raise ValueError('native decision option tokens are not token CE targets')
            continuations.add(node.token_ids[prompt:])
        if len(continuations) != len(options):
            raise ValueError('native decision options must have distinct continuations')
        if ('label' in value) == ('probabilities' in value):
            raise ValueError('native decision has exactly one of label or probabilities')
        if 'label' in value:
            label = value['label']
            if type(label) is not int or not 0 <= label < len(options):
                raise ValueError('native decision label must index one of its options')
            target = tuple(1. if index == label else 0. for index in range(len(options)))
        else:
            target = value['probabilities']
            if (not isinstance(target, list) or len(target) != len(options)
                    or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in target)
                    or not sum(target) > 0):
                raise ValueError('native decision probabilities are finite, nonnegative, one per option, '
                                 'with positive mass')
            total = float(sum(target))
            target = tuple(float(p) / total for p in target)
        if 'values' in value and (not isinstance(value['values'], list) or len(value['values']) != len(options)
                                  or any(not isinstance(item, str) for item in value['values'])):
            raise ValueError('native decision values are the option reply texts, one per option')
        result.append(GraphDecision(name, tuple(options), prompt, target, value['label_source']))
    return tuple(result)


def option_actions(example, decision):
    """Per option, the exact next-token queries of its continuation in the example's flat coordinates."""
    offsets = {node.name: offset for node, offset in zip(example.plan.invocations, example.plan.offsets)}
    nodes = {node.name: node for node in example.plan.invocations}
    result = []
    for option in decision.options:
        node, offset = nodes[option], offsets[option]
        result.append([{'query': offset + local - 1, 'token_id': int(node.token_ids[local]), 'temperature': 1.,
                        'allowed_token_ids': None, 'banned_token_ids': None}
                       for local in range(decision.prompt_tokens, len(node.token_ids))])
    return result


def tokenizer_renderer(tokenizer, *, model_type, history_reasoning_policy=None, prompt_rendering='full'):
    """The serving engine's exact chat renderer without a model, for CPU conversion (one implementation).

    Binds the engine's own rendering methods, so a converted decision has the same prompt and continuation tokens
    as ``/v1/neuralese/decide`` on an engine with the same tokenizer, template policy and prompt rendering.
    """
    from ..serve.chat import bind_history_reasoning, tool_call_format
    from ..serve.engine import Engine
    from ..data.prompt_rendering import PROMPT_RENDERINGS
    if prompt_rendering not in PROMPT_RENDERINGS:
        raise ValueError(f'unknown prompt_rendering {prompt_rendering!r}')

    class TokenizerRenderer:
        _template = Engine._template
        render_request = Engine.render_request
        _tokens = Engine._tokens
        _template_tokens = Engine._template_tokens
        specials = Engine.specials

    renderer = TokenizerRenderer()
    renderer.tokenizer = bind_history_reasoning(tokenizer, policy=history_reasoning_policy, model_type=model_type)
    renderer.prompt_rendering = prompt_rendering
    renderer.tool_call_format = tool_call_format(model_type)
    return renderer


def _decision_field(name, options, prompt_tokens, *, label=None, probabilities=None, label_source, values=None):
    value = {'name': name, 'options': list(options), 'prompt_tokens': prompt_tokens, 'label_source': label_source}
    if (label is None) == (probabilities is None):
        raise ValueError('a decision row needs exactly one of label or probabilities')
    if label is not None:
        value['label'] = label
    else:
        value['probabilities'] = list(probabilities)
    if values is not None:
        value['values'] = list(values)
    return value


def decision_capture_row(capture, *, id, split, source_groups, label_source, tokenizer_fingerprint, fetch_block,
                         label=None, probabilities=None, values=None):
    """A recorded ``natlang.native-decision-capture/1`` as a graph row; its native-state inputs are external states
    imported at their recorded positions (exact archived blocks, same actor). Decisions whose producers ran in the
    same episode are lowered with them by ``serve.native_recorded_graph`` instead."""
    from ..data.native_graph_schema import SCHEMA
    from ..serve.store import native_state_id
    if not isinstance(capture, dict) or capture.get('schema') != 'natlang.native-decision-capture/1':
        raise ValueError('decision row needs its actual native decision capture')
    own, imports, external = [], [], {}
    for segment in capture.get('prompt') or []:
        if segment.get('kind') == 'tokens':
            if any(segment.get('producing') or ()):
                raise ValueError('a decision prompt does not produce native state')
            own.extend(int(token) for token in segment['token_ids'])
        elif segment.get('kind') == 'native_state':
            if segment.get('adapter_state_consumption') is not None or 'semantic_adaptation' in segment:
                raise ValueError('adapter-state and semantic decision inputs lower through the recorded graph')
            block = fetch_block(segment['id'])
            name = f'input-{len(external)}'
            external[name] = {'id': native_state_id(block), 'token_count': len(block.token_ids)}
            imports.append({'at': len(own), 'source': name, 'start': 0, 'stop': len(block.token_ids),
                            **({'stop_gradient': True} if segment.get('stop_gradient') else {})})
        else:
            raise ValueError('unknown native decision prompt segment')
    continuations = capture.get('continuations')
    if not isinstance(continuations, list) or len(continuations) < 2:
        raise ValueError('decision capture needs its option continuations')
    nodes = [{'name': option_node_name('decision', index), 'token_ids': own + list(ids),
              'target_weights': [0.] * (len(own) + len(ids)), **({'imports': imports} if imports else {})}
             for index, ids in enumerate(continuations)]
    return {'schema': SCHEMA, 'id': id, 'split': split, 'source_groups': list(source_groups),
            'tokenizer_fingerprint': tokenizer_fingerprint, 'nodes': nodes, 'external_states': external,
            'decisions': [_decision_field('decision', [node['name'] for node in nodes], len(own), label=label,
                                          probabilities=probabilities, label_source=label_source, values=values)]}


def decision_graph_row(renderer, record, *, tokenizer_fingerprint):
    """One plain-text decision record as a native graph row with option nodes.

    ``record``: {id, split, source_groups, messages, tools?, options: [reply texts], label | probabilities,
    label_source, reasoning?}. Prompts with native-state inputs come from recorded captures
    (``serve.native_recorded_graph``), never from this text path. A record's ``reasoning`` (a reasoned readout's
    sampled reasoning) is prompt, as the scorer renders it (``decision_items(..., reasoning=)``); graph rows carry no
    token targets on decision prompts, so it conditions the choice and is not trained here.
    """
    from ..data.native_graph_schema import SCHEMA
    from ..serve.native_learning import decision_items
    prompt, continuations = decision_items(renderer, record['messages'], record.get('tools'), record['options'],
                                           bool(record.get('open_reply')), record.get('reasoning'))
    if any(kind != 'tok' for kind, _ in prompt):
        raise ValueError('a plain-text decision record cannot carry native-state inputs')
    ids = [int(value) for _, value in prompt]
    name = 'decision'
    nodes = [{'name': option_node_name(name, index), 'token_ids': ids + continuation,
              'target_weights': [0.] * (len(ids) + len(continuation))}
             for index, continuation in enumerate(continuations)]
    return {'schema': SCHEMA, 'id': record['id'], 'split': record['split'],
            'source_groups': list(record['source_groups']), 'tokenizer_fingerprint': tokenizer_fingerprint,
            'nodes': nodes, 'decisions': [_decision_field(name, [node['name'] for node in nodes], len(ids),
                label=record.get('label'), probabilities=record.get('probabilities'),
                label_source=record['label_source'], values=record['options'])]}
