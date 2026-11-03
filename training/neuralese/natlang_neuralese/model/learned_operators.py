"""Directly learned Neuralese operators: compaction, history carry and extraction (AGENTS.md 2026-11-03).

Owner intent: Neuralese compacts large values throughout the runtime (argument listings, inputs, tool and service
outputs, collections, documents), carries compressed state between model runs, and trains a bottleneck where a
Neuralese-prompted extractor feeds a parent that must use it. Each operator is a trained parameter that starts from
its natural-language prompt in the real rendered call and then morphs through direct learning on its own data.
Design and decision record: plans/neuralese/LEARNED_COMPACTION_OPERATORS.md.

An operator is one write site (the shared ``operator_site_messages``) with two learned row sets:

* **prompt rows**, one per token of the operator's prompt text, at the prompt's actual position in the rendered
  site. They are initialized from the pinned embedding rows of exactly those tokens, so at step 0 the site computes
  exactly what the text prompt computes (text initialization in the real call), and then train freely.
* **slot rows**, K positions after the operator's input inside a producing span. Initialized from the embeddings of
  the operator's slot text (its tokens cycled to K). The site's output is the native state of these K positions: a
  fixed-size bottleneck whose content is learned only through the consumers that import it. Writing it is a single
  prefill, never sampling, so the runtime can afford it for every large value.

Rows substitute token embeddings at declared positions (``apply``); everything else (positions, KV/hybrid state,
producer adapter, export and import) is the ordinary native graph machinery, so training, ablation and serving run
the same arithmetic. A graph row declares its operator positions in a top-level ``operators`` list; trainers that do
not load operator rows would see the operator's text tokens, i.e. its text-initialized form.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
from torch import nn

SCHEMA = 'natlang.learned-operator/1'
ROW_FIELD = 'operators'
SPAN_KINDS = ('prompt', 'slots')


@dataclass(frozen=True)
class Operator:
    name: str
    version: int
    # The natural-language prompt. Model-facing; changes require a version bump and a live measurement
    # (plans/MODEL_FACING_CHANGES.md).
    prompt: str
    # 'system': the prompt is the site's system text; 'user-suffix': a final user message after the input
    # conversation (history compaction keeps the conversation's own system/user messages in place).
    placement: str
    # Text whose tokens (cycled to ``slots``) initialize the slot rows and fill the slot positions' token IDs.
    slot_text: str
    slots: int
    # What one site reads (port-record source roles), and how its sources map to sites.
    source_roles: tuple[str, ...]
    sites: str  # 'one' (all sources in one site) or 'per-source' (fan-in: one producer per source)
    # Starting target of dedicated training examples (owner: about 20-30k each).
    data_target: int
    purpose: str
    # How a consumer is shown the imported state(s): open + state (+ separator + state ...) + close. For 'user-suffix'
    # operators the frame is its own user message after the first user message; otherwise it opens the first one.
    consumer_open: str = ''
    consumer_close: str = '\n\n'
    consumer_separator: str = '\n'


VALUE_PROMPT = (
    "Compress the value in the next message into the Neuralese slots that follow it. A later call reads those "
    "slots in place of the value, for the purpose the message states. Keep what that purpose needs: exact "
    "identifiers, paths, line numbers, names, numbers and values, how the value is organised, and where in it "
    "the rest can be found.")
PARAMS_PROMPT = (
    "Compress the argument in the next message into the Neuralese slots that follow it. The call that receives "
    "the argument shows these slots in its opening listing in place of the whole value, and reads the variable "
    "itself for exact details. Keep what the value is, how it is organised, and what in it bears on that call's "
    "instructions.")
HISTORY_PROMPT = (
    "Compress the conversation above into the Neuralese slots that follow. Work continues from these slots, the "
    "task statement and the most recent steps only. Keep what has been done and found, the exact files, "
    "identifiers, commands and values involved, what failed and why, and what is left to do.")
EXTRACT_PROMPT = (
    "Extract from the document in the next message what the stated task needs, into the Neuralese slots that "
    "follow it. A parent call combines these slots from several documents to complete its task and never sees the "
    "documents. Keep the facts the task needs exactly, and nothing else.")

OPERATORS = {operator.name: operator for operator in (
    Operator('compact-value', 1, VALUE_PROMPT, 'system',
             ' Exact names, numbers, paths and structure of the value, and where the rest is:', 64,
             ('tool_output', 'file', 'document', 'passage', 'record', 'schema', 'tool_doc'), 'one', 25_000,
             'Large tool/service outputs, documents and collections rendered into a call (renderValue).',
             'Compacted value: '),
    Operator('compact-params', 1, PARAMS_PROMPT, 'system',
             ' What the argument is, how it is organised, and what in it matters to the call:', 32,
             ('document', 'passage', 'file', 'record', 'schema', 'tool_output'), 'one', 25_000,
             "A call's large arguments in its opening listing (inputsListing).",
             'Compacted argument: '),
    Operator('compact-history', 1, HISTORY_PROMPT, 'user-suffix',
             ' Done, found, exact files, identifiers, commands and values, failures and what is left:', 128,
             ('trajectory',), 'one', 25_000,
             'Context carried between model runs (compact_history and the automatic near-limit compaction).',
             'Earlier steps of this session, compacted: ', '\n\nContinue from the most recent steps.'),
    Operator('extract', 1, EXTRACT_PROMPT, 'system',
             ' Facts this task needs from the document:', 32,
             ('document', 'passage', 'record', 'file', 'tool_output'), 'per-source', 30_000,
             'Neuralese-prompted extractor/summarizer whose state a parent lambda must use (bottleneck).',
             'Extracts from the documents, one per document:\n'),
)}


def operator_site_messages(name: str, *, instructions: str, source=None, context=(), prompt_part=None):
    """The operator's write site as serving messages, before the assistant reply (the slots).

    ``prompt_part`` stands in for the prompt text (a native-state placeholder part when rendering to tokens; the
    prompt text itself when None). ``source`` is the site's input text, or for 'user-suffix' operators the list of
    conversation messages compacted. The shared definition for the graph renderer and the serving endpoint.
    """
    spec = operator(name)
    prompt = spec.prompt if prompt_part is None else prompt_part
    def content(part):
        return part if isinstance(part, str) else [part]
    if spec.placement == 'system':
        if not isinstance(source, str):
            raise ValueError(f'operator {name} reads one text input')
        return [{'role': 'system', 'content': content(prompt)}, *[dict(message) for message in context],
                {'role': 'user', 'content': f'{instructions}\n\n{source}' if instructions else source}]
    if not isinstance(source, list):
        raise ValueError(f'operator {name} compacts a list of conversation messages')
    return [*[dict(message) for message in context], *[dict(message) for message in source],
            {'role': 'user', 'content': content(prompt)}]


def operator(name: str) -> Operator:
    try:
        return OPERATORS[name]
    except KeyError:
        raise ValueError(f'unknown learned operator {name!r}; registered: {sorted(OPERATORS)}') from None


def registry_document(names=None) -> dict:
    names = sorted(OPERATORS) if names is None else sorted(names)
    return {'schema': SCHEMA, 'operators': {name: {**asdict(operator(name)), 'source_roles': list(operator(name).source_roles)}
                                            for name in names}}


def registry_sha256(names=None) -> str:
    return hashlib.sha256(json.dumps(registry_document(names), sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def _token_ids(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)['input_ids']
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def bind_tokens(tokenizer, name: str, *, tokenizer_fingerprint: str) -> dict:
    """The exact token IDs an operator's rows sit on under one tokenizer.

    The prompt must tokenize standalone and decode back to itself: the site renderer inserts exactly these IDs at the
    prompt position, so the in-call token IDs are the standalone ones by construction.
    """
    spec = operator(name)
    prompt = _token_ids(tokenizer, spec.prompt)
    if not prompt or tokenizer.decode(prompt, skip_special_tokens=False) != spec.prompt:
        raise ValueError(f'operator {name} prompt does not round-trip through standalone tokenization')
    seed = _token_ids(tokenizer, spec.slot_text)
    if not seed or spec.slots < 1:
        raise ValueError(f'operator {name} needs a nonempty slot text and at least one slot')
    slots = [seed[index % len(seed)] for index in range(spec.slots)]
    return {'schema': SCHEMA, 'operator': name, 'version': spec.version, 'tokenizer_fingerprint': tokenizer_fingerprint,
            'prompt_token_ids': prompt, 'slot_token_ids': slots,
            'prompt_sha256': hashlib.sha256(spec.prompt.encode()).hexdigest(),
            'slot_text_sha256': hashlib.sha256(spec.slot_text.encode()).hexdigest()}


def _key(name):
    return name.replace('-', '_')


class OperatorRows(nn.Module):
    """Learned prompt and slot rows of the registered operators (FP32 masters, like the other learned rows)."""

    def __init__(self, bindings: dict, width: int, *, dtype=torch.float32, device=None):
        super().__init__()
        self.bindings = {}
        self.prompt = nn.ParameterDict()
        self.slots = nn.ParameterDict()
        for name, binding in sorted(bindings.items()):
            spec = operator(name)
            if binding.get('operator') != name or binding.get('version') != spec.version:
                raise ValueError(f'operator {name} binding does not match its registered version')
            if len(binding['slot_token_ids']) != spec.slots:
                raise ValueError(f'operator {name} binding has the wrong slot count')
            self.bindings[name] = {key: (list(value) if isinstance(value, list) else value)
                                   for key, value in binding.items()}
            self.prompt[_key(name)] = nn.Parameter(torch.zeros(len(binding['prompt_token_ids']), width,
                                                               dtype=dtype, device=device))
            self.slots[_key(name)] = nn.Parameter(torch.zeros(spec.slots, width, dtype=dtype, device=device))

    @classmethod
    def text_initialized(cls, embedding_weight, tokenizer, *, tokenizer_fingerprint, names=None, dtype=torch.float32):
        """Rows equal to the pinned embeddings of each operator's own text tokens (exact copies, widened)."""
        names = sorted(OPERATORS) if names is None else sorted(names)
        bindings = {name: bind_tokens(tokenizer, name, tokenizer_fingerprint=tokenizer_fingerprint) for name in names}
        rows = cls(bindings, embedding_weight.shape[1], dtype=dtype, device=embedding_weight.device)
        rows.reset_to_text(embedding_weight)
        return rows

    @torch.no_grad()
    def reset_to_text(self, embedding_weight):
        for name, binding in self.bindings.items():
            for table, field in ((self.prompt, 'prompt_token_ids'), (self.slots, 'slot_token_ids')):
                ids = torch.tensor(binding[field], dtype=torch.long, device=embedding_weight.device)
                table[_key(name)].copy_(embedding_weight[ids].to(table[_key(name)].dtype))

    def binding_document(self) -> dict:
        return {'schema': SCHEMA, 'registry_sha256': registry_sha256(self.bindings),
                'operators': {name: dict(binding) for name, binding in sorted(self.bindings.items())}}

    def rows(self, name: str, kind: str):
        if kind not in SPAN_KINDS:
            raise ValueError(f'operator span kind must be one of {SPAN_KINDS}')
        table = self.prompt if kind == 'prompt' else self.slots
        return table[_key(name)]

    def apply(self, embeddings, spans):
        """Substitute operator rows at resolved physical positions (out of place; gradients reach the rows)."""
        if not spans:
            return embeddings
        if embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError('operator rows apply to one [1, tokens, width] graph embedding tensor')
        out = embeddings.clone()
        for span in spans:
            rows = self.rows(span.operator, span.kind)
            if span.stop - span.start != rows.shape[0]:
                raise ValueError('operator span width differs from its learned rows')
            out[0, span.start:span.stop] = rows.to(device=out.device, dtype=out.dtype)
        return out


@dataclass(frozen=True)
class ResolvedSpan:
    operator: str
    kind: str
    node: str
    start: int  # physical graph position (plan offset + local start)
    stop: int


def resolve_spans(row: dict, plan, bindings: dict) -> tuple[ResolvedSpan, ...]:
    """Validate a row's declared operator positions against its compiled plan and the bound token IDs."""
    declared = row.get(ROW_FIELD) or []
    if not isinstance(declared, list):
        raise ValueError('graph operators must be a list of declared spans')
    names = [node.name for node in plan.invocations]
    resolved, taken = [], set()
    for entry in declared:
        name, kind = entry.get('operator'), entry.get('kind')
        if name not in bindings:
            raise ValueError(f'graph declares operator {name!r}, which has no loaded rows')
        binding = bindings[name]
        if entry.get('version') != binding['version'] or kind not in SPAN_KINDS:
            raise ValueError(f'graph operator {name} span has an unbound version or kind')
        if entry.get('node') not in names:
            raise ValueError(f'graph operator {name} names an unknown node')
        index = names.index(entry['node'])
        node = plan.invocations[index]
        start, stop = entry.get('start'), entry.get('stop')
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(node.token_ids):
            raise ValueError(f'graph operator {name} span is outside its node')
        expected = binding['prompt_token_ids' if kind == 'prompt' else 'slot_token_ids']
        if list(node.token_ids[start:stop]) != list(expected):
            raise ValueError(f'graph operator {name} {kind} tokens differ from the bound operator tokens')
        if kind == 'slots' and not any(a <= start and stop <= b for a, b in node.producing):
            raise ValueError(f'graph operator {name} slots must lie inside a producing span')
        positions = range(plan.offsets[index] + start, plan.offsets[index] + stop)
        if taken.intersection(positions):
            raise ValueError('graph operator spans overlap')
        taken.update(positions)
        resolved.append(ResolvedSpan(name, kind, entry['node'], positions.start, positions.stop))
    return tuple(resolved)


def operator_embeddings(model, rows: OperatorRows | None, plan, spans):
    """Graph input embeddings with operator rows substituted (``model.embed`` of the plan's own tokens otherwise)."""
    ids = plan.tensor([token for node in plan.invocations for token in node.token_ids], model.input_rows.device)
    raw = model.embed(ids[None])
    return raw if rows is None or not spans else rows.apply(raw, spans)


def save_rows(rows: OperatorRows, path) -> None:
    """Rows with their exact token bindings (a standalone artifact for ablation and serving probes)."""
    torch.save({'schema': SCHEMA, 'bindings': rows.binding_document(),
                'state': {key: value.detach().cpu() for key, value in rows.state_dict().items()}}, path)


def load_rows(path, *, device=None) -> OperatorRows:
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('schema') != SCHEMA:
        raise ValueError('not a learned operator rows artifact')
    bindings = saved['bindings']['operators']
    if saved['bindings'].get('registry_sha256') != registry_sha256(bindings):
        raise ValueError('operator rows were saved under a different operator registry; rebind them explicitly')
    width = next(iter(saved['state'].values())).shape[1]
    rows = OperatorRows(bindings, width, device=device)
    rows.load_state_dict(saved['state'])
    return rows
