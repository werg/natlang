"""Learned-operator cases -> native graphs (plans/neuralese/COMPACTION_OPERATOR_CONVERTERS.md §1).

One `natlang.port-record/1` case for one learned operator (model/learned_operators.py) becomes:

* producer node(s) ``op<i>``: the operator write site (``operator_site_messages``) rendered with the serving chat
  template and its generation prompt, the operator prompt's token IDs at the prompt position, then
  ``produce_open, seed tokens, produce_close``. The seed is the source's supplied gold/checked span
  (``sources[i].meta.operator_seed``) or the shared deterministic mechanical seed (data/operator_seeds.py), clipped
  to the operator's span cap. ``producing`` covers the seed; the seed tokens and the closing marker carry CE weight 1
  (the trainer's declared seed schedule anneals it); the site itself has zero weight.
* the consumer node: the case's consumer conversation with the operator's consumer frame, where each producer's
  state is imported between ``consume_open``/``consume_close``; the target reply is the only supervised span.
* a text-cut sibling: the same consumer reading the runtime's current text cut of each source (ts-host
  native/cutoff.ts ``cutText`` at its 2,000-character floor) instead of the state. It is the baseline the operator
  must beat (eval/neuralese_ablation.py ``--text-cut``).

Only gold/checked targets are supervised; admission and the protection union are the caller's
(scripts/neuralese_data/build_operator_graphs.py). Nothing is truncated: a case over the token cap is rejected.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json

from ..model.learned_operators import operator, operator_site_messages, registry_sha256
from .operator_seeds import seed_for
from .native_graph_schema import SCHEMA, parse_graph
from .port_record_graph import _serving_messages, _target_message, _target_reply

CONVERSION = 'natlang.operator-graph-conversion/1'
GUIDANCE_CONVERSION = 'natlang.guidance-operator-graph-conversion/1'
SHOWN_CHARS = 2000  # ts-host native/cutoff.ts SHOWN_CHARS
IMITATION_LABELS = {'gold', 'checked'}


class OperatorCaseError(ValueError):
    """A case that cannot become an operator graph; the message is its omission reason."""


def cut_text(text: str, budget: int = SHOWN_CHARS) -> str:
    """Port of ts-host native/cutoff.ts ``cutText`` (no holder): head and tail with the cut-off note between."""
    if len(text) <= budget:
        return text
    head = budget * 3 // 4
    tail = len(text) - (budget - head)
    head_break = text.rfind('\n', 0, head + 1)
    tail_break = text.find('\n', tail)
    if head_break > head - 200:
        head = head_break
    if 0 <= tail_break < tail + 200:
        tail = tail_break + 1
    return f'{text[:head]}\n<<cut off: {tail - head} of {len(text)} characters not shown>>\n{text[tail:]}'


def _placeholder(tag: str) -> str:
    digest = hashlib.sha256(tag.encode()).digest()
    return 'nzs1_' + base64.b32encode(digest).decode('ascii').lower().rstrip('=')


def source_text(source: dict) -> str:
    if isinstance(source.get('text'), str):
        return source['text']
    if isinstance(source.get('messages'), list):
        return json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    raise OperatorCaseError('source-without-text')


class OperatorGraphRenderer:
    """Renders operator cases with one expanded tokenizer and the bound operator token IDs."""

    def __init__(self, tokenizer, *, role_ids, tokenizer_fingerprint, bindings, max_tokens=32768):
        from ..serve.chat import bind_history_reasoning
        self.tokenizer, self.role_ids, self.fingerprint = tokenizer, tuple(role_ids), tokenizer_fingerprint
        self.bindings, self.max_tokens = bindings, max_tokens
        bound = getattr(tokenizer, 'natlang_history_reasoning', {})
        bind_history_reasoning(tokenizer, policy=bound.get('policy'))
        self.specials = tuple(tokenizer.all_special_tokens)

    def _ids(self, text, escaped=False):
        ids = self.tokenizer(text, add_special_tokens=False, split_special_tokens=escaped)['input_ids']
        return list(ids[0] if ids and isinstance(ids[0], list) else ids)

    def _render(self, messages, tools, splices, *, generation):
        """Token IDs of messages whose native-state placeholder parts are replaced by ``splices[id]``."""
        from ..serve.chat import render_messages, split_escaped
        rendered = render_messages(messages, tools, lambda turns, schemas: self.tokenizer.apply_chat_template(
            turns, tools=schemas or None, tokenize=False, add_generation_prompt=generation), specials=self.specials)
        if sorted(rendered.blocks) != sorted(splices) or len(set(rendered.blocks)) != len(rendered.blocks):
            raise OperatorCaseError('render-placeholders: the template dropped or duplicated a placeholder')
        ids, spans = [], {}
        for segment in rendered.segments:
            if isinstance(segment, int):
                block = rendered.blocks[segment]
                spans[block] = (len(ids), len(ids) + len(splices[block]))
                ids.extend(splices[block])
            else:
                for run, escaped in split_escaped(segment, rendered.escape_nonce):
                    ids.extend(self._ids(run, escaped))
        return ids, spans

    def site(self, name, *, instructions, source, context=(), tools=None):
        """Token IDs of one operator write site through ``produce_open``: (ids, prompt span).

        The single definition of the site for training graphs and the serving endpoint (engine.native_operator),
        which generates the producer span after it. ``source`` is text for 'system' operators and serving messages
        for 'user-suffix' operators.
        """
        spec, binding = operator(name), self.bindings[name]
        prompt_id = _placeholder(f'operator-prompt:{name}')
        messages = operator_site_messages(name, instructions=instructions, source=source, context=context,
                                          prompt_part={'type': 'native_state', 'id': prompt_id})
        ids, spans = self._render(messages, tools if spec.placement != 'system' else None,
                                  {prompt_id: list(binding['prompt_token_ids'])}, generation=True)
        ids.append(self.role_ids[0])
        return ids, spans[prompt_id]

    def seed(self, name, record, source):
        """(seed token IDs, seed record) of one source: its supplied seed, else the shared mechanical seed."""
        spec = operator(name)
        seed = seed_for(name, record, source)
        ids = [token for token in self._ids(seed['text'], escaped=True) if token not in self.role_ids]
        if not ids:
            raise OperatorCaseError('seed-empty')
        clipped = len(ids) > spec.seed_cap
        return ids[:spec.seed_cap], {'provenance': seed['provenance'], 'origin': seed.get('origin'),
                                     'tokens': min(len(ids), spec.seed_cap), 'clipped': clipped,
                                     'text_sha256': hashlib.sha256(seed['text'].encode()).hexdigest()}

    def _producer(self, name, index, record, source, tools):
        spec = operator(name)
        context = _serving_messages(record['writer'].get('context') or [])
        if spec.placement == 'system':
            site_source = source_text(source)
        else:
            if not isinstance(source.get('messages'), list):
                raise OperatorCaseError('history-source-without-messages')
            site_source = _serving_messages(source['messages'])
        ids, (prompt_start, prompt_stop) = self.site(
            name, instructions=record['writer']['instructions'], source=site_source, context=context, tools=tools)
        seed_ids, seed = self.seed(name, record, source)
        seed_start = len(ids)
        weights = [0.] * len(ids) + [1.] * (len(seed_ids) + 1)
        ids = [*ids, *seed_ids, self.role_ids[1]]
        seed_stop = seed_start + len(seed_ids)
        node = {'name': f'op{index}', 'token_ids': ids, 'target_weights': weights,
                'producing': [[seed_start, seed_stop]]}
        declared = [{'operator': name, 'version': spec.version, 'node': node['name'], 'kind': 'prompt',
                     'start': prompt_start, 'stop': prompt_stop},
                    {'operator': name, 'version': spec.version, 'node': node['name'], 'kind': 'seed',
                     'start': seed_start, 'stop': seed_stop, 'provenance': seed['provenance']}]
        return node, declared, (seed_start, seed_stop), seed

    def _consumer_messages(self, name, record, parts):
        """The consumer conversation with ``parts`` (the framed states or texts) placed per the operator."""
        spec = operator(name)
        messages = _serving_messages(record['consumer']['context'])
        first = next((i for i, message in enumerate(messages) if message.get('role') == 'user'), None)
        if first is None:
            raise OperatorCaseError('consumer-without-user-message')
        framed = [{'type': 'text', 'text': spec.consumer_open}] if spec.consumer_open and not spec.frame_each else []
        for position, part in enumerate(parts):
            if position:
                framed.append({'type': 'text', 'text': spec.consumer_separator})
            if spec.frame_each and spec.consumer_open and not part.get('inline'):
                framed.append({'type': 'text', 'text': spec.consumer_open})
            framed.append({key: value for key, value in part.items() if key != 'inline'})
        if spec.placement == 'system':
            framed.append({'type': 'text', 'text': spec.consumer_close})
            old = messages[first].get('content')
            if isinstance(old, str):
                messages[first]['content'] = [*framed, {'type': 'text', 'text': old}]
            elif isinstance(old, list):
                messages[first]['content'] = [*framed, *copy.deepcopy(old)]
            else:
                messages[first]['content'] = framed
        else:
            framed.append({'type': 'text', 'text': spec.consumer_close})
            messages.insert(first + 1, {'role': 'user', 'content': framed})
        # Adjacent text parts are joined so the template sees ordinary strings around each state.
        for message in messages:
            content = message.get('content')
            if isinstance(content, list):
                merged = []
                for part in content:
                    if part.get('type') == 'text' and merged and merged[-1].get('type') == 'text':
                        merged[-1] = {'type': 'text', 'text': merged[-1]['text'] + part['text']}
                    else:
                        merged.append(dict(part))
                message['content'] = (merged[0]['text'] if len(merged) == 1 and merged[0]['type'] == 'text'
                                      else merged)
        return messages

    def render(self, record: dict, name: str, *, corpus_id: str, source_row_sha256: str, source_ref: dict):
        """(graph row, text-cut row) for one case, or OperatorCaseError with the omission reason."""
        if name == 'guidance':
            return self.render_guidance(record, corpus_id=corpus_id, source_row_sha256=source_row_sha256,
                                        source_ref=source_ref)
        spec = operator(name)
        if record.get('outcome', {}).get('label') not in IMITATION_LABELS:
            raise OperatorCaseError('target-not-imitation-label')
        if 'sources' not in record['consumer'].get('withheld', []):
            raise OperatorCaseError('sources-not-withheld')
        sources = record['sources']
        if spec.sites == 'one' and len(sources) != 1:
            raise OperatorCaseError(f'site-count: {len(sources)} sources for a one-site operator')
        tools = record['consumer'].get('tools') or None
        target = _target_message(_RowView(record))
        # Sources the runtime would not cut are shown in full (no site); the others each get an operator site.
        inline = [bool(spec.inline_below_chars) and len(source_text(source)) < spec.inline_below_chars
                  for source in sources]
        if all(inline):
            raise OperatorCaseError('no-compacted-source: every source is below the runtime cut threshold')
        compacted = [source for source, shown in zip(sources, inline) if not shown]
        nodes, declared, slots, seeds = [], [], [], []
        for index, source in enumerate(compacted):
            node, spans, slot, seed = self._producer(name, index, record, source, tools)
            nodes.append(node); declared.extend(spans); slots.append(slot); seeds.append(seed)
        try:
            target_ids = _target_reply(self.tokenizer, target, tools)
        except ValueError as error:
            raise OperatorCaseError(f'target-render: {error}') from None
        state_ids = [_placeholder(f'operator-state:{name}:{index}') for index in range(len(compacted))]
        consume_open, consume_close = self.role_ids[2], self.role_ids[3]
        states = iter(state_ids)
        messages = self._consumer_messages(name, record, [
            {'type': 'text', 'text': source_text(source), 'inline': True} if shown
            else {'type': 'native_state', 'id': next(states)} for source, shown in zip(sources, inline)])
        ids, spans = self._render(messages, tools, {state: [consume_open, consume_close] for state in state_ids},
                                  generation=True)
        imports = [{'at': spans[state][0] + 1, 'source': f'op{index}', 'start': slot[0], 'stop': slot[1]}
                   for index, (state, slot) in enumerate(zip(state_ids, slots))]
        context = len(ids)
        ids.extend(target_ids)
        consumer = {'name': 'consumer', 'token_ids': ids,
                    'target_weights': [0.] * context + [1.] * len(target_ids), 'imports': imports}
        total = sum(len(node['token_ids']) for node in nodes) + len(ids)
        if total > self.max_tokens:
            raise OperatorCaseError(f'over-token-cap: {total} > {self.max_tokens}')
        split = 'test' if record['split'] in ('test', 'validation') else 'train'
        groups = sorted(set(record['split_groups']))
        graph_id = f'operator:{name}:{source_row_sha256}'
        conversion = {'schema': CONVERSION, 'operator': name, 'operator_version': spec.version,
                      'registry_sha256': registry_sha256([name]), 'corpus_id': corpus_id,
                      'binding_sha256': hashlib.sha256(json.dumps(self.bindings[name], sort_keys=True).encode()).hexdigest(),
                      'source_row_sha256': source_row_sha256, 'source_ref': source_ref, 'original_id': record['id'],
                      'original_split': record['split'], 'family': record.get('family'),
                      'outcome_label': record['outcome']['label'], 'outcome_checked': record['outcome'].get('checked'),
                      'lineage': record.get('lineage'), 'license': record.get('license'),
                      'source_chars': [len(source_text(source)) for source in compacted],
                      'inline_source_chars': [len(source_text(source)) for source, shown in zip(sources, inline) if shown],
                      'source_tokens': [len(node['token_ids']) for node in nodes],
                      'seeds': seeds, 'span_tokens': [stop - start for start, stop in slots],
                      'target_tokens': len(target_ids),
                      'context_weight': 0.0}
        graph = {'schema': SCHEMA, 'id': graph_id, 'split': split, 'source_groups': groups,
                 'tokenizer_fingerprint': self.fingerprint, 'nodes': [*nodes, consumer], 'operators': declared,
                 'operator_conversion': conversion}
        cut_messages = self._consumer_messages(name, record, [
            {'type': 'text', 'text': source_text(source), 'inline': True} if shown
            else {'type': 'text', 'text': cut_text(source_text(source))} for source, shown in zip(sources, inline)])
        cut_ids, _ = self._render(cut_messages, tools, {}, generation=True)
        cut_context = len(cut_ids)
        cut_ids.extend(target_ids)
        text_cut = {'schema': SCHEMA, 'id': 'text-cut:' + graph_id, 'text_cut_of': graph_id, 'split': split,
                    'source_groups': groups, 'tokenizer_fingerprint': self.fingerprint,
                    'nodes': [{'name': 'consumer', 'token_ids': cut_ids,
                               'target_weights': [0.] * cut_context + [1.] * len(target_ids)}],
                    'operator_conversion': {**conversion, 'variant': 'text-cut', 'shown_chars': SHOWN_CHARS}}
        for row in (graph, text_cut):
            try:
                parse_graph(row, tokenizer_fingerprint=self.fingerprint)
            except ValueError as error:
                raise OperatorCaseError(f'graph-invalid: {error}') from None
        return graph, text_cut


    def privileged_rendering(self):
        """The engine's guidance-turn rendering for this tokenizer (rl/privileged_context.PrivilegedRendering)."""
        from types import SimpleNamespace
        from .render import Renderer
        from ..rl.privileged_context import PrivilegedRendering
        from ..serve.chat import tokenizer_model_type
        # The serving engine's own construction (serve/engine.py privileged rendering), so the digest matches.
        renderer = Renderer(self.tokenizer, SimpleNamespace(open_id=self.role_ids[0], close_id=self.role_ids[1]))
        return PrivilegedRendering.from_renderer(renderer, template=tokenizer_model_type(self.tokenizer),
                                                 tokenizer_fingerprint=self.fingerprint,
                                                 consume_open=self.role_ids[2], consume_close=self.role_ids[3])

    def render_guidance(self, case: dict, *, corpus_id: str, source_row_sha256: str, source_ref: dict):
        """(graph row, no-guidance-baseline rows) of one ``natlang.guidance-case/1`` (data/guidance_cases.py).

        Producer ``op0``: the recorded writeGuidance call (its request messages and tools) with the operator prompt
        rows on the writer's instruction text, then ``produce_open``, the seed (the checked text-writer guidance,
        else the mechanical seed of the facts it was given) and ``produce_close``; seed and close carry CE weight 1.
        Consumers ``consumer<i>``: each consuming invocation's recorded student-form first turn with the engine's
        guidance turn (header, then the imported state between consume markers) spliced in after its leading
        system turn, exactly as ``guidance_plan`` places it; its checked reply is the only supervised span. The
        sibling is the same consumers with the facts as the engine's text turn (the runtime's text fallback): the
        baseline the operator must beat.
        """
        from .operator_seeds import guidance_seed, supplied_seed
        from .guidance_cases import SCHEMA as CASE_SCHEMA, check_case
        if case.get('schema') != CASE_SCHEMA:
            raise OperatorCaseError('not-a-guidance-case')
        try:
            check_case(case)
        except ValueError as error:
            raise OperatorCaseError(f'case-invalid: {error}') from None
        if case['outcome'].get('label') not in IMITATION_LABELS:
            raise OperatorCaseError('target-not-imitation-label')
        spec, binding = operator('guidance'), self.bindings['guidance']
        writer = case['writer']
        site_text = '\n'.join(message['content'] for message in writer['messages'] if isinstance(message.get('content'), str))
        if any(fact['text'] not in site_text for fact in writer['facts']):
            raise OperatorCaseError('writer-site-lacks-facts: a credited fact is not in the recorded writer call')
        tools = writer.get('tools') or None
        try:
            ids, (prompt_start, prompt_stop) = self.site('guidance', instructions='', tools=tools,
                                                         source=_serving_messages(writer['messages']))
        except ValueError as error:
            raise OperatorCaseError(f'writer-site: {error}') from None
        seed = supplied_seed({'meta': {'operator_seed': case['guidance']}}) if case.get('guidance') else None
        if seed is None:
            try:
                seed = guidance_seed(writer['facts'])
            except ValueError as error:
                raise OperatorCaseError(str(error)) from None
        seed_ids = [token for token in self._ids(seed['text'], escaped=True) if token not in self.role_ids]
        if not seed_ids:
            raise OperatorCaseError('seed-empty')
        clipped = len(seed_ids) > spec.seed_cap
        seed_ids = seed_ids[:spec.seed_cap]
        seed_start, seed_stop = len(ids), len(ids) + len(seed_ids)
        producer = {'name': 'op0', 'token_ids': [*ids, *seed_ids, self.role_ids[1]],
                    'target_weights': [0.] * len(ids) + [1.] * (len(seed_ids) + 1), 'producing': [[seed_start, seed_stop]]}
        declared = [{'operator': 'guidance', 'version': spec.version, 'node': 'op0', 'kind': 'prompt',
                     'start': prompt_start, 'stop': prompt_stop},
                    {'operator': 'guidance', 'version': spec.version, 'node': 'op0', 'kind': 'seed',
                     'start': seed_start, 'stop': seed_stop, 'provenance': seed['provenance']}]
        rendering = self.privileged_rendering()
        state_turn, import_offset = rendering.state_turn()
        # The engine's text fallback packs whole facts into its token budget (privileged_context.guidance_plan).
        shown = []
        for fact in writer['facts']:
            if len(rendering.turn([*shown, fact['text']])) <= rendering.budget_tokens:
                shown.append(fact['text'])
        text_turn = rendering.turn(shown) if shown else []
        consumers, baselines, target_tokens = [], [], []
        for index, consumer in enumerate(case['consumers']):
            context, _ = self._render(_serving_messages(consumer['context']), consumer.get('tools') or None, {},
                                      generation=True)
            try:
                insert_at, _ = rendering.insertion_point(context)
                target_ids = _target_reply(self.tokenizer, _target_message(_RowView({'target': consumer['target']})),
                                           consumer.get('tools') or None)
            except ValueError as error:
                raise OperatorCaseError(f'consumer-render: {error}') from None
            own = [*context[:insert_at], *state_turn, *context[insert_at:]]
            consumers.append({'name': f'consumer{index}', 'token_ids': [*own, *target_ids],
                              'target_weights': [0.] * len(own) + [1.] * len(target_ids),
                              'imports': [{'at': insert_at + import_offset, 'source': 'op0',
                                           'start': seed_start, 'stop': seed_stop}]})
            cut = [*context[:insert_at], *text_turn, *context[insert_at:]]
            baselines.append({'name': f'consumer{index}', 'token_ids': [*cut, *target_ids],
                              'target_weights': [0.] * len(cut) + [1.] * len(target_ids)})
            target_tokens.append(len(target_ids))
        total = len(producer['token_ids']) + sum(len(node['token_ids']) for node in consumers)
        if total > self.max_tokens:
            raise OperatorCaseError(f'over-token-cap: {total} > {self.max_tokens}')
        split = 'test' if case['split'] in ('test', 'validation') else 'train'
        groups = sorted(set(case['split_groups']))
        graph_id = f'operator:guidance:{source_row_sha256}'
        conversion = {'schema': GUIDANCE_CONVERSION, 'operator': 'guidance', 'operator_version': spec.version,
                      'registry_sha256': registry_sha256(['guidance']), 'corpus_id': corpus_id,
                      'binding_sha256': hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
                      'rendering': rendering.declaration(), 'source_row_sha256': source_row_sha256,
                      'source_ref': source_ref, 'original_id': case['id'], 'original_split': case['split'],
                      'family': case.get('family'), 'outcome_label': case['outcome']['label'],
                      'outcome_checked': case['outcome'].get('checked'), 'lineage': case.get('lineage'),
                      'seeds': [{'provenance': seed['provenance'], 'origin': seed.get('origin'),
                                 'tokens': len(seed_ids), 'clipped': clipped,
                                 'text_sha256': hashlib.sha256(seed['text'].encode()).hexdigest()}],
                      'facts': [{'id': fact['id'], 'kind': fact['kind']} for fact in writer['facts']],
                      'span_tokens': [seed_stop - seed_start], 'site_tokens': len(ids),
                      'target_tokens': target_tokens, 'context_weight': 0.0}
        graph = {'schema': SCHEMA, 'id': graph_id, 'split': split, 'source_groups': groups,
                 'tokenizer_fingerprint': self.fingerprint, 'nodes': [producer, *consumers], 'operators': declared,
                 'operator_conversion': conversion}
        text_cut = {'schema': SCHEMA, 'id': 'text-cut:' + graph_id, 'text_cut_of': graph_id, 'split': split,
                    'source_groups': groups, 'tokenizer_fingerprint': self.fingerprint, 'nodes': baselines,
                    'operator_conversion': {**conversion, 'variant': 'facts-text-turn'}}
        for row in (graph, text_cut):
            try:
                parse_graph(row, tokenizer_fingerprint=self.fingerprint)
            except ValueError as error:
                raise OperatorCaseError(f'graph-invalid: {error}') from None
        return graph, text_cut


class _RowView:
    """The minimal PortRecordRow surface `_target_message` reads."""

    def __init__(self, record):
        self.raw = record
