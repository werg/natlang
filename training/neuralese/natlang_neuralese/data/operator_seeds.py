"""Seeds of learned-operator producer spans (plans/neuralese/LEARNED_COMPACTION_OPERATORS.md, owner 2026-11-03).

A learned operator writes a variable-length producer span. Training starts that span as a text summary related to the
consumer's target, supervised with an annealed CE weight, so the imported state carries input-specific content from
step 0; the consumer's outcome gradients then morph it. The seed is never invented:

* ``gold`` / ``checked``: a dataset or code-verified span the converter supplies (supporting facts, reference
  summaries, answer sentences, relevant spans, exact code-derived recall);
* ``mechanical:<method>``: a deterministic extraction from the source itself, computed here, when no such span
  exists (a source is never left out for lacking one).

Converters put a supplied seed in ``sources[i].meta.operator_seed = {"text", "provenance", "origin"}``; the renderer
(data/operator_graph.py) uses it, else ``mechanical_seed``. One shared implementation for every converter and the
renderer; every function is deterministic.
"""
from __future__ import annotations

import json
import re

PROVENANCES = ('gold', 'checked')
_SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])|\n+')
_WORD = re.compile(r'[A-Za-z0-9_]+')
_IDENT = re.compile(r'(?:[\w.-]+/)+[\w.-]+(?::\d+)?|\b[A-Za-z_][\w]*\.[A-Za-z_][\w.]*\b|\b\d[\d.,:%-]*\b|https?://\S+|'
                    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|`[^`\n]{1,60}`')
STOP = frozenset('the a an of to in on and or for is are was were be by with as at from that this it its which who '
                 'what when where how why do does did not no yes'.split())


def budget_chars(source_chars: int, *, floor: int = 200, ceiling: int = 2400, share: float = 0.08) -> int:
    """A seed length budget scaled to the input size, between ``floor`` and ``ceiling`` characters."""
    return int(max(floor, min(ceiling, source_chars * share)))


def _clip(text: str, budget: int) -> str:
    text = text.strip()
    if len(text) <= budget:
        return text
    cut = text.rfind('\n', 0, budget)
    return text[:cut if cut > budget // 2 else budget].rstrip()


def sentences(text: str) -> list[str]:
    return [piece.strip() for piece in _SENTENCE.split(text) if piece and piece.strip()]


def _words(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text) if word.lower() not in STOP}


def answer_lines(text: str, answers, budget: int) -> str | None:
    """Sentences/lines containing any answer string (case-insensitive), in source order."""
    wanted = [answer.lower() for answer in answers if isinstance(answer, str) and answer.strip()]
    if not wanted:
        return None
    hits = [piece for piece in sentences(text) if any(answer in piece.lower() for answer in wanted)]
    return _clip('\n'.join(dict.fromkeys(hits)), budget) if hits else None


def overlap_sentences(text: str, reference: str, budget: int, *, limit: int = 12) -> str | None:
    """The sentences with the largest lexical overlap with ``reference`` (question or target), in source order."""
    wanted = _words(reference)
    if not wanted:
        return None
    scored = []
    for index, piece in enumerate(dict.fromkeys(sentences(text))):
        overlap = len(_words(piece) & wanted)
        if overlap:
            scored.append((-overlap, index, piece))
    if not scored:
        return None
    chosen = sorted(sorted(scored)[:limit], key=lambda item: item[1])
    return _clip('\n'.join(piece for _, _, piece in chosen), budget)


def identifiers(text: str, budget: int, *, limit: int = 64) -> str | None:
    """The exact identifiers, paths, locations, numbers and names present, in first-seen order."""
    seen = list(dict.fromkeys(match.group(0).strip('`') for match in _IDENT.finditer(text)))[:limit]
    return _clip(', '.join(seen), budget) if seen else None


def outline(text: str, budget: int) -> str:
    """Structure outline plus head and tail: JSON keys/shape, line count, headings and definitions."""
    lines = text.splitlines()
    parts = [f'{len(lines)} lines, {len(text)} characters.']
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict):
        parts.append('JSON object with keys: ' + ', '.join(list(value)[:40]))
    elif isinstance(value, list):
        kinds = {type(item).__name__ for item in value[:200]}
        keys = list(dict.fromkeys(key for item in value[:200] if isinstance(item, dict) for key in item))[:30]
        parts.append(f'JSON array of {len(value)} items ({", ".join(sorted(kinds))})' + (f' with keys: {", ".join(keys)}' if keys else ''))
    heads = [line.strip() for line in lines if re.match(r'\s*(#{1,6}\s|def |class |function |export |interface |\[\w)', line)]
    if heads:
        parts.append('Sections and definitions: ' + '; '.join(heads[:30]))
    half = max(80, (budget - sum(len(part) for part in parts)) // 2)
    parts.append('Starts: ' + text[:half].strip())
    if len(text) > 2 * half:
        parts.append('Ends: ' + text[-half:].strip())
    return _clip('\n'.join(parts), budget)


def history_outline(messages: list, budget: int) -> str:
    """A mechanical account of a conversation stretch: actions taken, paths touched, errors, and its last steps."""
    actions, paths, errors = [], [], []
    for message in messages:
        for call in message.get('tool_calls') or []:
            function = call.get('function') or call
            arguments = str(function.get('arguments') or '')
            actions.append(f"{function.get('name')}({arguments[:80]})")
        content = message.get('content') if isinstance(message.get('content'), str) else ''
        paths.extend(re.findall(r'(?:/[\w.-]+){2,}(?::\d+)?', content))
        errors.extend(line.strip() for line in content.splitlines()
                      if re.search(r'\b(Error|Exception|Traceback|FAILED|failed)\b', line))
    last = next((message.get('content') for message in reversed(messages)
                 if message.get('role') == 'assistant' and isinstance(message.get('content'), str) and message['content'].strip()), '')
    parts = [f'{len(messages)} messages, {len(actions)} tool calls.']
    if actions:
        parts.append('Actions: ' + '; '.join(actions[-20:]))
    if paths:
        parts.append('Paths: ' + ', '.join(list(dict.fromkeys(paths))[:30]))
    if errors:
        parts.append('Errors: ' + ' | '.join(list(dict.fromkeys(errors))[:8]))
    if last:
        parts.append('Last step: ' + last.strip()[:600])
    return _clip('\n'.join(parts), budget)


def mechanical_seed(operator: str, source: dict, *, question: str = '', answers=(), target: str = '') -> dict:
    """The deterministic seed of one source when no gold/checked span exists: {text, provenance}.

    Chosen per value type: a conversation gets a history outline; otherwise the answer-bearing lines, else the
    sentences overlapping the question/target, else (for exact-recall operators) the identifiers present, else the
    structure outline with head and tail. The length budget scales with the input size.
    """
    if isinstance(source.get('messages'), list):
        text = json.dumps(source['messages'], ensure_ascii=False)
        return {'text': history_outline(source['messages'], budget_chars(len(text), ceiling=4000)),
                'provenance': 'mechanical:history-outline'}
    text = source.get('text') or ''
    budget = budget_chars(len(text))
    seed = answer_lines(text, answers, budget)
    if seed:
        return {'text': seed, 'provenance': 'mechanical:answer-lines'}
    reference = ' '.join(part for part in (question, target) if part)
    seed = overlap_sentences(text, reference, budget) if operator in ('extract', 'compact-params') else None
    if seed:
        return {'text': seed, 'provenance': 'mechanical:overlap-sentences'}
    if operator == 'compact-value':
        seed = identifiers(text, budget)
        structure = outline(text, budget // 2 if seed else budget)
        return {'text': _clip((seed + '\n' + structure) if seed else structure, budget),
                'provenance': 'mechanical:identifiers-outline' if seed else 'mechanical:outline'}
    return {'text': outline(text, budget), 'provenance': 'mechanical:outline'}


def chunk_summary_seed(text: str, question: str = '') -> dict:
    """A deterministic summary of one chunk of a long text: {text, provenance, parts}.

    The question-selected sentences when a question is known (the spans the consumer needs most), then the exact
    identifiers, paths, numbers and names present, then the structure outline (headings, definitions, head/tail).
    The budget scales with the chunk size like every mechanical seed; a source is never left without one.
    """
    budget = budget_chars(len(text))
    chosen = []
    for part, seed in (('question-overlap', overlap_sentences(text, question, budget // 2) if question else None),
                       ('identifiers', identifiers(text, budget // 4)),
                       ('outline', outline(text, max(200, budget // 2)))):
        if seed:
            chosen.append((part, seed))
    return {'text': _clip('\n'.join(seed for _, seed in chosen), budget),
            'provenance': 'mechanical:chunk-summary', 'parts': [part for part, _ in chosen]}


# Fact kinds the mechanical guidance seed leads with, and how it introduces each. Feedback/failure kinds describe a
# mistaken attempt: the writer is told never to repeat a mistaken result (writeGuidance.nl step 4), so the mechanical
# seed does not copy them either.
GUIDANCE_FACT_LEADS = {
    'expected_result': 'A correct result', 'expected_stop': 'A correct stop', 'expected_files': 'Files a correct result has',
    'expected_effects': 'Effects a correct result has', 'child_value': 'A correct value for this call',
    'reference_solution': 'A way to get there', 'minimum_sequence': 'Steps that get there',
    'decisive': 'What decides it', 'teacher_trajectory': 'How a verified run did it',
    'fit_hindsight': 'What worked'}


def guidance_seed(facts: list) -> dict:
    """The deterministic guidance of one writer site when no checked guidance exists: {text, provenance}.

    Built only from the verified facts the writer was given, in their selected order: one numbered line per usable fact
    (its kind's lead, then the fact text), clipped to a budget scaled to the facts' size. Input-specific by construction
    (the facts are the call's own); feedback and failure facts are left out (they describe a mistaken attempt).
    """
    usable = [fact for fact in facts if isinstance(fact, dict) and fact.get('kind') in GUIDANCE_FACT_LEADS
              and isinstance(fact.get('text'), str) and fact['text'].strip()]
    if not usable:
        raise ValueError('guidance-seed-without-usable-facts')
    budget = budget_chars(sum(len(fact['text']) for fact in usable), floor=200, ceiling=2000, share=0.5)
    lines = [f"{number}. {GUIDANCE_FACT_LEADS[fact['kind']]}: {fact['text'].strip()}"
             for number, fact in enumerate(usable, 1)]
    return {'text': _clip('\n'.join(lines), budget), 'provenance': 'mechanical:guidance-facts'}


def supplied_seed(source: dict) -> dict | None:
    """A converter-supplied gold/checked seed, validated; None when absent."""
    seed = (source.get('meta') or {}).get('operator_seed')
    if seed is None:
        return None
    if (not isinstance(seed, dict) or not isinstance(seed.get('text'), str) or not seed['text'].strip()
            or not (seed.get('provenance') in PROVENANCES or str(seed.get('provenance', '')).startswith('mechanical:'))):
        raise ValueError('operator_seed needs text and a gold, checked or mechanical:<method> provenance')
    return seed


def _last_user_text(messages) -> str:
    for message in reversed(messages):
        if message.get('role') == 'user':
            content = message.get('content')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return ' '.join(part.get('text', '') for part in content if isinstance(part, dict))
    return ''


def target_text(record: dict) -> str:
    """A port record target's own text, for seed selection: text/choice/sql values, message content, else JSON."""
    target = record.get('target') or {}
    value = target.get('value')
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get('content'), str) and value['content'].strip():
        return value['content']
    if isinstance(value, dict) and value.get('tool_calls'):
        return json.dumps(value['tool_calls'], ensure_ascii=False, sort_keys=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else ''


def seed_for(operator: str, record: dict, source: dict) -> dict:
    """The seed of one source of a case: its supplied seed, else the mechanical seed from the case's question/target."""
    seed = supplied_seed(source)
    if seed is not None:
        return seed
    target = target_text(record)
    question = '\n'.join(part for part in ((record.get('writer') or {}).get('instructions') or '',
                                           _last_user_text((record.get('consumer') or {}).get('context') or [])) if part)
    answers = (target,) if 0 < len(target.strip()) <= 200 else ()
    return {**mechanical_seed(operator, source, question=question, answers=answers, target=target),
            'origin': 'natlang_neuralese.data.operator_seeds'}
