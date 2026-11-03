"""Directly learned Neuralese operators: compaction, history carry and extraction (AGENTS.md 2026-11-03).

Owner intent: Neuralese compacts large values throughout the runtime (argument listings, inputs, tool and service
outputs, collections, documents), carries compressed state between model runs, and trains a bottleneck where a
Neuralese-prompted extractor feeds a parent that must use it. Design and decision record:
plans/neuralese/LEARNED_COMPACTION_OPERATORS.md.

An operator is one write site (the shared ``operator_site_messages``) followed by a **generated producer span**:

* **prompt rows**, one per token of the operator's prompt text, at the prompt's actual position in the rendered
  site, initialized from the pinned embedding rows of exactly those tokens (step 0 computes exactly what the text
  prompt computes) and then trained freely.
* **the producer span** between ``produce_open`` and ``produce_close``: variable length up to the operator's
  ``seed_cap`` tokens. Training starts it as a *seed*, a text summary related to the consumer's target (gold or
  checked spans, else a deterministic mechanical extraction, data/operator_seeds.py), supervised by CE whose weight
  anneals on a declared schedule while the consumer's outcome gradients flow through the span's state (the
  morph). Serving generates the span greedily and archives its native state. Owner correction 2026-11-03: the
  earlier fixed-K slot rows seeded from a generic slot text carried no input-specific content at step 0 and are
  gone; a single-prefill slot writer distilled from trained spans is an optional later addition.

Prompt rows substitute token embeddings at declared positions (``apply``); everything else (positions, KV/hybrid
state, producer adapter, export and import) is the ordinary native graph machinery, so training, ablation and serving
run the same arithmetic. A graph row declares its operator positions in a top-level ``operators`` list: ``prompt``
spans (row substitution) and ``seed`` spans (the supervised producer span whose CE weight the schedule scales).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
from torch import nn

SCHEMA = 'natlang.learned-operator/1'
ROW_FIELD = 'operators'
SPAN_KINDS = ('prompt', 'seed')


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
    # Longest producer span in tokens: seeds are clipped to it and serving stops generating at it.
    seed_cap: int
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
    "Write Neuralese that stands in for the value in the next message. A later call reads it in place of the value, "
    "for the purpose the message states. Keep what that purpose needs: exact identifiers, paths, line numbers, "
    "names, numbers and values, how the value is organised, and where in it the rest can be found.")
PARAMS_PROMPT = (
    "Write Neuralese that stands in for the argument in the next message. The call that receives the argument "
    "shows it in its opening listing in place of the whole value, and reads the variable itself for exact details. "
    "Keep what the value is, how it is organised, and what in it bears on that call's instructions.")
HISTORY_PROMPT = (
    "Write Neuralese that stands in for the conversation above. Work continues from it, the task statement and the "
    "most recent steps only. Keep what has been done and found, the exact files, identifiers, commands and values "
    "involved, what failed and why, and what is left to do.")
EXTRACT_PROMPT = (
    "Write Neuralese holding what the stated task needs from the document in the next message. A parent call "
    "combines this from several documents to complete its task and never sees the documents. Keep the facts the "
    "task needs exactly, and nothing else.")

OPERATORS = {operator.name: operator for operator in (
    Operator('compact-value', 2, VALUE_PROMPT, 'system', 512,
             ('tool_output', 'file', 'document', 'passage', 'record', 'schema', 'tool_doc'), 'one', 25_000,
             'Large tool/service outputs, documents and collections rendered into a call (renderValue).',
             'Compacted value: '),
    Operator('compact-params', 2, PARAMS_PROMPT, 'system', 384,
             ('document', 'passage', 'file', 'record', 'schema', 'tool_output'), 'one', 25_000,
             "A call's large arguments in its opening listing (inputsListing).",
             'Compacted argument: '),
    Operator('compact-history', 2, HISTORY_PROMPT, 'user-suffix', 1024,
             ('trajectory',), 'one', 25_000,
             'Context carried between model runs (compact_history and the automatic near-limit compaction).',
             'Earlier steps of this session, compacted: ', '\n\nContinue from the most recent steps.'),
    Operator('extract', 2, EXTRACT_PROMPT, 'system', 256,
             ('document', 'passage', 'record', 'file', 'tool_output'), 'per-source', 30_000,
             'Neuralese-prompted extractor/summarizer whose state a parent lambda must use (bottleneck).',
             'Extracts from the documents, one per document:\n'),
)}


def operator_site_messages(name: str, *, instructions: str, source=None, context=(), prompt_part=None):
    """The operator's write site as serving messages, before the assistant reply (the producer span).

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
    if spec.seed_cap < 1:
        raise ValueError(f'operator {name} needs a positive producer span cap')
    return {'schema': SCHEMA, 'operator': name, 'version': spec.version, 'tokenizer_fingerprint': tokenizer_fingerprint,
            'prompt_token_ids': prompt, 'seed_cap': spec.seed_cap,
            'prompt_sha256': hashlib.sha256(spec.prompt.encode()).hexdigest()}


def _key(name):
    return name.replace('-', '_')


class OperatorRows(nn.Module):
    """Learned prompt rows of the registered operators (FP32 masters, like the other learned rows)."""

    def __init__(self, bindings: dict, width: int, *, dtype=torch.float32, device=None):
        super().__init__()
        self.bindings = {}
        self.prompt = nn.ParameterDict()
        for name, binding in sorted(bindings.items()):
            spec = operator(name)
            if binding.get('operator') != name or binding.get('version') != spec.version:
                raise ValueError(f'operator {name} binding does not match its registered version')
            if binding.get('seed_cap') != spec.seed_cap:
                raise ValueError(f'operator {name} binding has a different producer span cap')
            self.bindings[name] = {key: (list(value) if isinstance(value, list) else value)
                                   for key, value in binding.items()}
            self.prompt[_key(name)] = nn.Parameter(torch.zeros(len(binding['prompt_token_ids']), width,
                                                               dtype=dtype, device=device))

    @classmethod
    def text_initialized(cls, embedding_weight, tokenizer, *, tokenizer_fingerprint, names=None, dtype=torch.float32):
        """Rows equal to the pinned embeddings of each operator's prompt tokens (exact copies, widened)."""
        names = sorted(OPERATORS) if names is None else sorted(names)
        bindings = {name: bind_tokens(tokenizer, name, tokenizer_fingerprint=tokenizer_fingerprint) for name in names}
        rows = cls(bindings, embedding_weight.shape[1], dtype=dtype, device=embedding_weight.device)
        rows.reset_to_text(embedding_weight)
        return rows

    @torch.no_grad()
    def reset_to_text(self, embedding_weight):
        for name, binding in self.bindings.items():
            ids = torch.tensor(binding['prompt_token_ids'], dtype=torch.long, device=embedding_weight.device)
            self.prompt[_key(name)].copy_(embedding_weight[ids].to(self.prompt[_key(name)].dtype))

    def binding_document(self) -> dict:
        return {'schema': SCHEMA, 'registry_sha256': registry_sha256(self.bindings),
                'operators': {name: dict(binding) for name, binding in sorted(self.bindings.items())}}

    def rows(self, name: str, kind: str = 'prompt'):
        if kind != 'prompt':
            raise ValueError('only operator prompt spans carry learned rows; seed spans are ordinary tokens')
        return self.prompt[_key(name)]

    def apply(self, embeddings, spans):
        """Substitute prompt rows at resolved physical positions (out of place; gradients reach the rows)."""
        spans = [span for span in spans if span.kind == 'prompt']
        if not spans:
            return embeddings
        if embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError('operator rows apply to one [1, tokens, width] graph embedding tensor')
        out = embeddings.clone()
        for span in spans:
            rows = self.rows(span.operator)
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
    return resolve_declared(row.get(ROW_FIELD) or [], plan, bindings)


def resolve_declared(declared, plan, bindings: dict, *, offset: int = 0) -> tuple[ResolvedSpan, ...]:
    """``resolve_spans`` over a declared list; ``offset`` shifts physical positions (a graph packed after others)."""
    declared = list(declared or [])
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
        if kind == 'prompt' and list(node.token_ids[start:stop]) != list(binding['prompt_token_ids']):
            raise ValueError(f'graph operator {name} prompt tokens differ from the bound operator tokens')
        if kind == 'seed':
            if not any(a <= start and stop <= b for a, b in node.producing):
                raise ValueError(f'graph operator {name} seed must lie inside a producing span')
            if stop - start > binding['seed_cap']:
                raise ValueError(f'graph operator {name} seed exceeds the operator span cap')
        positions = range(plan.offsets[index] + start, plan.offsets[index] + stop)
        if taken.intersection(positions):
            raise ValueError('graph operator spans overlap')
        taken.update(positions)
        resolved.append(ResolvedSpan(name, kind, entry['node'], offset + positions.start, offset + positions.stop))
    return tuple(resolved)


def attach_operator_rows(model, tokenizer, *, tokenizer_fingerprint, names=None):
    """Give a native model text-initialized operator rows (a recorded learned-row addition), or keep its own.

    Returns the upgrade record: the operators added now (empty when the model already carries exactly these
    bindings), the registry digest and the parameter names a resumed optimizer must accept as new.
    """
    names = sorted(OPERATORS) if names is None else sorted(names)
    wanted = {name: bind_tokens(tokenizer, name, tokenizer_fingerprint=tokenizer_fingerprint) for name in names}
    existing = getattr(model, 'operator_rows', None)
    if existing is not None:
        if existing.bindings != wanted:
            raise ValueError('model operator rows are bound to different operators or tokens; declare an explicit rebinding')
        return {'schema': SCHEMA, 'added': [], 'registry_sha256': registry_sha256(names), 'parameter_names': []}
    dtype = model.input_rows.dtype
    rows = OperatorRows(wanted, model.port.embedding_weight.shape[1], dtype=dtype, device=model.port.embedding_weight.device)
    rows.reset_to_text(model.port.embedding_weight)
    model.operator_rows = rows
    parameter_names = ['interface.operator_rows.' + name for name, _ in rows.named_parameters()]
    return {'schema': SCHEMA, 'added': names, 'registry_sha256': registry_sha256(names),
            'parameter_names': parameter_names, 'dtype': str(dtype), 'initialization': 'text-embedding-rows'}


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


SEED_SCHEDULE = 'natlang.operator-seed-schedule/1'
# Full seed CE at adoption, annealed over 2,000 updates to a 0.1 floor: the floor keeps the generator that writes the
# span at serving time anchored to readable, input-specific content while consumer gradients reshape its state.
DEFAULT_SEED_SCHEDULE = {'start': 1.0, 'end': 0.1, 'updates': 2000}


def seed_schedule(spec) -> dict | None:
    """Validate a declared seed-CE schedule: ``{start, end, updates}``, linear in optimizer updates, then ``end``.

    The seed is a starting point, not a fixed target: its CE weight anneals while the consumer's outcome gradients
    keep flowing through the producer span (owner correction 2026-11-03). None means no scaling (weight 1).
    """
    if spec is None:
        return None
    if (not isinstance(spec, dict) or set(spec) - {'start', 'end', 'updates'}
            or any(type(spec.get(key)) not in (int, float) or not 0 <= spec[key] <= 1 for key in ('start', 'end'))
            or type(spec.get('updates')) is not int or spec['updates'] < 1):
        raise ValueError('operator_seed_schedule is {start, end, updates}: weights in [0, 1] and a positive update count')
    return {'schema': SEED_SCHEDULE, 'start': float(spec['start']), 'end': float(spec['end']), 'updates': spec['updates']}


def seed_factor(schedule: dict | None, update: int) -> float:
    """The seed CE weight at an optimizer update counted from the schedule's adoption."""
    if schedule is None:
        return 1.
    progress = min(1., max(0, update) / schedule['updates'])
    return schedule['start'] + (schedule['end'] - schedule['start']) * progress


def seed_weighted(example, factor: float):
    """``example`` with its declared seed spans' CE weights (seed tokens and the closing marker) scaled by ``factor``.

    Every other weight, the graph and its imports are unchanged; an example without seed spans is returned as is.
    """
    seeds = [entry for entry in getattr(example, 'operators', ()) if entry.get('kind') == 'seed']
    if not seeds or factor == 1.:
        return example
    import dataclasses
    names = [node.name for node in example.plan.invocations]
    weights = list(example.target_weights)
    for entry in seeds:
        index = names.index(entry['node'])
        node, offset = example.plan.invocations[index], example.plan.offsets[index]
        for local in range(entry['start'], min(entry['stop'] + 1, len(node.token_ids))):
            weights[offset + local] *= factor
    return dataclasses.replace(example, target_weights=tuple(weights))
