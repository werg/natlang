"""T_self: consumers reading the producer's own emitted tokens as crisp text (owner decision 2026-10-13).

The primary end-to-end comparison (train/e2e_acceptance.py) sets consumers reading the producer's Neuralese values
(V) against the same consumers reading the text the same producer generates itself (T_self), not the gold text: the
fair runtime comparison, since at serving a text hand-off carries the producer's own tokens. Gold T stays as the
reported upper reference.

How a self text is made, per producer and variant (greedy, and sampled at temperature 0.6 with a fixed seed per
producer so it is reproducible):
1. The producer's crisp rendering (every soft part as its text, every value it reads as text: its own parents' self
   texts down to --write-depth, gold beyond, as the written path writes them) plus its native write prefix
   (``site_prefix``: the model's own reply through the opening of the written argument).
2. Free-running decoding (not teacher-forced) with the writer's own token readout, lm_head(norm(TCP(h))) over the
   write segment (``token_choice_logits``, as serve/direct_write.py chooses), each chosen token fed back as its
   ordinary embedding.
3. Stop at the native value boundary: the closing quote of a string argument (for an inline-code writer, the first
   character of the code after its site, the template literal's backtick), or the end of a complete value for an
   ``unknown`` (JSON) write; else at a stop token or at twice the gold value's token count (``cap_hit``).
4. The span decodes to the value whose native text it is (checked by re-rendering it through the template,
   ``exact``); the consumer's crisp rendering then reads that value where the gold one stood (``substitute_sources``,
   the same read parts the gold T renders from).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re

import torch

MODES = (('greedy', 0.0), ('t06', 0.6))
_MISSING = object()
_SCALAR = re.compile(r'(?:true|false|null|True|False|None|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')
_SCALAR_CONTINUES = set('0123456789.eE+-_') | set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')


def string_end(text: str, quote: str) -> int | None:
    """Index of the first unescaped ``quote`` in ``text`` (the end of a string argument's body), or None."""
    index = text.find(quote)
    while index >= 0:
        slashes = len(text[:index]) - len(text[:index].rstrip('\\'))
        if slashes % 2 == 0:
            return index
        index = text.find(quote, index + 1)
    return None


def value_end(text: str, final: bool = False) -> int | None:
    """End of the complete value starting ``text`` (after leading whitespace), or None while it is incomplete. A bare
    number or literal is complete once a character follows that cannot continue it, or at the end when ``final``."""
    lead = len(text) - len(text.lstrip())
    body = text[lead:]
    if not body:
        return None
    if body[0] in '\'"':
        end = string_end(body[1:], body[0])
        return None if end is None else lead + end + 2
    if body[0] in '{[':
        try:
            _, end = json.JSONDecoder().raw_decode(body)
        except ValueError:
            return None
        return lead + end
    match = _SCALAR.match(body)
    if match and ((match.end() < len(body) and body[match.end()] not in _SCALAR_CONTINUES)
                  or (final and match.end() == len(body))):
        return lead + match.end()
    return None


def decode_value(span: str, value_type: str, quote: str, native) -> tuple[object, bool]:
    """(value, exact): the value whose native text (``native(value)``) is ``span``. Candidates are the template's
    Python-style and JSON spellings; the first that re-renders to ``span`` exactly wins, else the first that parses
    (exact False), else (_MISSING, False)."""
    if value_type == 'string':
        literals = [lambda: ast.literal_eval(quote + span + quote)]
        if quote == '"':
            literals.insert(0, lambda: json.loads('"' + span + '"'))
    else:
        literals = [lambda: json.loads(span.strip()), lambda: ast.literal_eval(span.strip())]
    parsed = []
    for literal in literals:
        try:
            value = literal()
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if value_type == 'string' and not isinstance(value, str):
            continue
        parsed.append(value)
        try:
            if native(value) == span:
                return value, True
        except (ValueError, TypeError):
            pass
    return (parsed[0], False) if parsed else (_MISSING, False)


def serialize_like(gold_source: str, value) -> str:
    """``value`` as JSON in the gold source's own spelling (compact or spaced), so only the content differs."""
    styles = ({'separators': (',', ':'), 'ensure_ascii': False}, {'ensure_ascii': False},
              {'separators': (',', ':')}, {})
    try:
        gold = json.loads(gold_source)
        style = next((s for s in styles if json.dumps(gold, **s) == gold_source), styles[0])
    except ValueError:
        style = styles[0]
    return json.dumps(value, **style)


def value_source(span: str, value_type: str, quote: str, native, gold_source: str) -> tuple[str, bool]:
    """(source text the consumer reads, exact decode) for a generated native span."""
    value, exact = decode_value(span, value_type, quote, native)
    if value is _MISSING:
        return (span if value_type == 'string' else span.strip()), False
    if value_type == 'string':
        return value, exact
    try:
        return serialize_like(gold_source, value), exact
    except (TypeError, ValueError):
        return span.strip(), False


def generate_span(next_logits, advance, decode, *, stop_ids, end_of, cap: int, temperature: float = 0.0,
                  generator: torch.Generator | None = None) -> dict:
    """Free-running decoding until ``end_of(text, final)`` finds the value boundary, a stop token, or ``cap`` tokens.

    ``next_logits()`` is the readout at the newest state ([V]); ``advance(token)`` feeds the chosen token; ``decode``
    maps ids to text. Greedy at temperature 0, else sampled from softmax(logits / temperature) with ``generator``."""
    ids, end = [], 'cap'
    while len(ids) < cap:
        logits = next_logits().float()
        if temperature:
            probs = torch.softmax(logits / temperature, -1).cpu()
            token = int(torch.multinomial(probs, 1, generator=generator))
        else:
            token = int(logits.argmax(-1))
        if token in stop_ids:
            end = 'stop_token'
            break
        ids.append(token)
        text = decode(ids)
        found = end_of(text, False)
        if found is not None:
            return {'ids': ids, 'text': text, 'span': text[:found], 'end': 'boundary', 'cap_hit': False}
        advance(token)
    text = decode(ids)
    found = end_of(text, True)
    return {'ids': ids, 'text': text, 'span': text if found is None else text[:found],
            'end': end if found is None else 'boundary', 'cap_hit': end == 'cap'}


def substitute_sources(record: dict, overrides: dict[str, str]) -> dict:
    """``record`` with every read of a name in ``overrides`` (content parts, tool-call argument parts, ``$write`` sites
    in its history) carrying that text as its source: the gold crisp rendering's own hook (``render`` shows a read's
    source), so T_self and gold T render identically apart from the values."""
    if not overrides:
        return record
    from .write_sites import extract_write_sites_from_calls

    def visit(value):
        if isinstance(value, dict):
            if value.get('type') == 'read' and value.get('name') in overrides:
                return {**value, 'source': overrides[value['name']]}
            return {key: visit(child) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child) for child in value]
        return value

    def write_value(value):
        if isinstance(value, dict) and isinstance(value.get('$write'), dict):
            site = value['$write']
            if site.get('name') not in overrides:
                return value
            source = overrides[site['name']]
            if site.get('source_encoding') == 'json' or site.get('type') == 'Neuralese<unknown>':
                try:
                    json.loads(source)
                except ValueError:
                    source = json.dumps(source, ensure_ascii=False)
            return {**value, '$write': {**site, 'source': source}}
        if isinstance(value, dict):
            return {key: write_value(child) for key, child in value.items()}
        if isinstance(value, list):
            return [write_value(child) for child in value]
        return value

    messages = visit(record['messages'])
    for message in messages:
        calls = message.get('tool_calls') or []
        for index, call in enumerate(calls):
            if 'neuralese_code' in call:
                if any(site.name in overrides for site in extract_write_sites_from_calls([call])):
                    raise ValueError('self text cannot replace an inline-code writer in the reader history')
                continue
            arguments = (call.get('function') or {}).get('arguments')
            if isinstance(arguments, str) and '"$write"' in arguments:
                calls[index] = {**call, 'function': {**call['function'],
                                                     'arguments': json.dumps(write_value(json.loads(arguments)),
                                                                             ensure_ascii=False)}}
    return {**record, 'messages': messages}


def producer_seed(producer_id: str, name: str, mode: str) -> int:
    digest = hashlib.sha256(f'{producer_id}\0{name}\0{mode}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)


class SelfText:
    """Self texts of one evaluation pass (fixed weights): memoized per (name, depth, mode)."""

    def __init__(self, *, engine, session, producers: dict, texts: dict, write_depth: int, site_prefix,
                 token_choice_projection=None):
        self.engine, self.session, self.producers, self.texts = engine, session, producers, texts
        self.write_depth, self.site_prefix, self.tcp = write_depth, site_prefix, token_choice_projection
        self.memo = {}

    def _value_type(self, producer):
        from .trajectories import write_value_type
        return 'string' if self.engine.heads.read_markers else write_value_type(producer)

    def _native(self, producer, name, value_type):
        """(native, quote, delimiter): the value's native spelling (as ``producer_source`` makes the gold one), the
        quote of its string argument, and the character that ends a string value natively: that quote, or for an
        inline-code writer the first character of the code after its site (the template literal's backtick)."""
        from ..serve.chat import write_reply, write_value_text
        from .trajectories import write_site, write_site_arguments, write_value_path
        from .write_sites import extract_write_sites
        tool, before, argument, _ = write_site(producer)
        path = write_value_path(producer, name)
        arguments = write_site_arguments(producer, name) if len(path) > 1 else before
        argument_path = path if len(path) > 1 else None
        template = lambda m, g: self.engine.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=g)
        native = lambda value: write_value_text(template, tool, arguments, argument, value, value_type,
                                                argument_path=argument_path)
        opening = write_reply(template, tool, arguments, argument, 'string', argument_path)[0]
        quote = opening[-1] if opening and opening[-1] in '\'"' else '"'
        delimiter = quote
        site = next((s for s in extract_write_sites(producer) if s.name == name), None)
        if site is not None and site.source_kind == 'inline-code-sidecar':
            code, before_site = site.arguments.get('code'), site.inline_prefix
            if (isinstance(code, str) and isinstance(before_site, str) and code.startswith(before_site)
                    and code[len(before_site):].startswith(site.source)):
                rest = code[len(before_site) + len(site.source):]
                spelled = native(rest) if rest else ''
                delimiter = spelled[0] if spelled else quote
        return native, quote, delimiter

    def _context(self, producer, name, overrides):
        from ..serve.chat import render_messages
        from .trajectories import authenticated_capture_context_augmentation, crisp_messages, handover_notes
        shown = substitute_sources(producer, overrides)
        messages = crisp_messages(shown['messages'], self.texts, handover_notes(shown))
        capture, _ = authenticated_capture_context_augmentation(producer, name)
        if capture is not None:
            messages = [*messages, capture]
        prompt = render_messages(messages, producer.get('tools'), self.engine._template, self.engine.specials)
        items = self.session._items(prompt.segments, prompt.blocks, prompt.escape_nonce)
        items += [('tok', t) for t in self.engine._tokens(self.site_prefix(producer))]
        return self.session._embed_items(items, {})

    def text(self, name: str, mode: str, temperature: float, depth: int = 1, visiting: tuple = ()) -> dict:
        key = name, depth, mode
        if key in self.memo:
            return self.memo[key]
        from .trajectories import handover_notes, reads, target_writes
        producer = self.producers[name]
        overrides = {}
        if depth < self.write_depth:
            parents = (reads(producer) | set(handover_notes(producer))) - target_writes(producer)
            for parent in sorted(parents):
                if parent in self.producers and parent not in visiting and parent != name:
                    overrides[parent] = self.text(parent, mode, temperature, depth + 1,
                                                  visiting + (name,))['source']
        value_type = self._value_type(producer)
        native, quote, delimiter = self._native(producer, name, value_type)
        gold_source = handover_notes(producer)[name]
        gold_value = json.loads(gold_source) if value_type == 'unknown' else gold_source
        gold_tokens = len(self.engine._tokens(native(gold_value)))
        generated = self._generate(self._context(producer, name, overrides), value_type, delimiter,
                                   cap=max(1, 2 * gold_tokens), temperature=temperature,
                                   seed=producer_seed(str(producer.get('id')), name, mode))
        source, exact = value_source(generated['span'], value_type, quote, native, gold_source)
        self.memo[key] = result = {'source': source, 'exact': exact, 'end': generated['end'],
                                   'cap_hit': generated['cap_hit'], 'tokens': len(generated['ids']),
                                   'gold_tokens': gold_tokens}
        return result

    @torch.no_grad()
    def _generate(self, context, value_type, delimiter, *, cap, temperature, seed):
        from ..model.neuralese_projection import causal_conv_receptive_field, token_choice_logits
        from .execution import prefill_write_context
        backbone = self.engine.backbone
        pre = prefill_write_context(backbone, self.engine.heads, context)
        history, state = [pre.top], {'cache': pre.cache}
        limit = causal_conv_receptive_field(self.tcp) if self.tcp is not None else 1

        def next_logits():
            hidden = torch.stack(history[-limit:], 1)
            segments = torch.zeros(hidden.shape[:2], dtype=torch.long, device=hidden.device)
            return token_choice_logits(backbone, hidden, self.tcp, segments, last_only=True)[0, -1]

        def advance(token):
            embedded = backbone.embed(torch.tensor([[token]], device=pre.top.device)).to(backbone.embedding_weight.dtype)
            final, state['cache'] = backbone.run_layers(embedded, range(backbone.num_layers), state['cache'])
            history.append(final[:, 0])
            del history[:-limit]

        end_of = ((lambda text, final: string_end(text, delimiter)) if value_type == 'string' else value_end)
        generator = torch.Generator().manual_seed(seed) if temperature else None
        return generate_span(next_logits, advance,
                             lambda ids: self.engine.tokenizer.decode(ids, skip_special_tokens=False),
                             stop_ids=set(self.engine.stop_ids), end_of=end_of, cap=cap,
                             temperature=temperature, generator=generator)

    def scores(self, record: dict, names, crisp_loss) -> dict:
        """The reader's crisp CE per variant with each read producer value as that producer's self text."""
        targets = sorted(name for name in names if name in self.producers)
        out = {'texts': {}}
        for mode, temperature in MODES:
            made = {name: self.text(name, mode, temperature) for name in targets}
            out[mode] = crisp_loss(substitute_sources(record, {n: m['source'] for n, m in made.items()}))
            out['texts'][mode] = [{'name': n, **{k: m[k] for k in ('tokens', 'gold_tokens', 'end', 'exact', 'cap_hit')}}
                                  for n, m in made.items()]
        out['cap_hit'] = any(m['cap_hit'] for rows in out['texts'].values() for m in rows)
        return out
