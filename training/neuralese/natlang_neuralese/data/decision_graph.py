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


def decision_prompt(replies):
    return ('Answer this call now, without tools. Reply with exactly one of these JSON values and nothing else: '
            + ', '.join(replies) + '.')


def decision_messages(opening, values, *, guidance=''):
    """The readout's messages for a finite-typed call (agent.ts decisionReadout): the decision system prompt (plus
    program guidance), the call's opening without its system turn and the answer request over the JSON values."""
    import json
    replies = [json.dumps(value, ensure_ascii=False, separators=(',', ':')) for value in values]
    return ([{'role': 'system', 'content': DECISION_SYSTEM_PROMPT + guidance}, *opening[1:],
             {'role': 'user', 'content': decision_prompt(replies)}], replies)
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
    label_source}. Prompts with native-state inputs come from recorded captures (``serve.native_recorded_graph``),
    never from this text path.
    """
    from ..data.native_graph_schema import SCHEMA
    from ..serve.native_learning import decision_items
    prompt, continuations = decision_items(renderer, record['messages'], record.get('tools'), record['options'])
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
