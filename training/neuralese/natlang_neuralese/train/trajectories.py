"""Trajectory training on Neuralese-converted natlang records (DECISIONS.md 40, 41; S5 §3, first step).

Input: records written by ts-host/scripts/neuralese-convert-trajectories.mjs and its pieces file. Every `soft` part
(the runtime's prompt pieces, older system-prompt versions, program guidance) is a trainable soft parameter,
initialised from its text: from a system-prompt bank where one is given and the piece is in it (`prompt:<id>`),
otherwise encoded from the text in one pass through the port (`encode_text`). Each training step teacher-forces the record's target turn (cross-entropy on
its tokens) with the soft parameters as gradient leaves, optionally with a LoRA on the backbone. The trained
parameters are saved as a bank of the current runtime's pieces (`system-prompts.nz`) and as all soft parameters by
name (`soft-params.pt`).

Written values (`--handover written`, `--view written`): handover notes, child calls' results and listing views are written by the
model through the port's differentiable write procedure (S3 `unroll_write`), afresh at every step, and enter their
readers as gradient leaves. The readers' losses therefore train the writer: the payload carries gradients into the
content projection, the sketch recurrence, the LoRA and every soft parameter of the write site (view's body, the
producing record's prompts), so a note or view learns to hold what its readers need. A note is
written from its producing record (the record whose target is the `compact_history` call), soft-rendered, at the note
argument (the reply forced to the model's own rendering of `compact_history(note='`, chat.call_reply, then the
write: the same cut the server's template readout makes). A child call's result (converter: child results) is
written the same way from the child's final record at `return_result(status='success', value='` and read where the
caller's eval output shows it: the recurrence of calling a function, retrieving its value and splicing it into the
caller's trajectory, trained across the run's chunks (the child's record and every caller record that reads it) in
one graph. Writes nest: a producer's own context reads the values it was given written afresh too, to
`--write-depth` levels (a caller reads a child's result whose child read a grandchild's); deeper ones are crisp. A
view part (a listing value, or a tool output in harness-bench records) is the Neuralese instance of the builtin
`view(value, instructions)`: written at view's template write site by its plan (view.py: view's body as the system
text, the reply forced to `return_result(status="success", value="` and the write, as the servers' `/v1/neuralese/view`
does), chunked when the value exceeds `--view-window` tokens, every chunk write and the combining write
differentiable. With `--tokens-per-vector R` a write is sized from the crisp text it stands for (the note's text, the
listing preview a view replaces): ceil(tokens / R) vectors, no stop decision, and the stop head is trained on that
boundary (`--stop-weight`). This is the simple training regime; sizes are never required at inference, where the stop
head decides unless a caller passes a length hint. Otherwise stop decisions are sampled and trained by a policy gradient with reward −(reader loss + λ·length)
against a running baseline (`--stop-pg λ`); without it they are detached and the length is the stop head's choice. With the crisp modes, notes are rendered as the crisp note and
views as the listing's preview. A view part may carry its own `instructions` (what it is written for, for example an
agent's intent at the tool call whose output it views) and `note` (how the reader gets the whole value); otherwise the
view is written for the receiving call (view.listing_instructions) and the variable note applies; a part marked
`faithful` is written without instructions (`view(x)`: the view stage's reconstruction records, data/view_records.py).
The view stage's `--purpose-contrast w` adds, for records with `view_stage.contrast_instructions`, a hinge on the
reader's CE with the same value viewed for a partner purpose (`--purpose-margin`). Records converted
before the rename carry `digest` parts, which are rejected with the conversion to run. Records that read written values add a self-distillation term (weight `--distill`)
from the same model given the crisp note and preview.

Evaluation on held-out records (split `test`): mean target cross-entropy with the system prompt as crisp text, with the
soft parameters as initialised, and as trained. With written values, the held-out readers are also scored with the
values written for them against values written for another reader of different values (`written-init`,
`written-trained`: written, shuffled, the fraction where written is better); `-train` on the first --eval training
readers (whether readers use the written content at all, when few held-out readers fit).

Usage: python -m natlang_neuralese.train.trajectories --records converted.jsonl --pieces pieces.jsonl --out DIR
         [--heads CKPT --bank system-prompts.nz --steps 500 --batch 4 --lr 1e-3 --rank 0 --lora-lr 2e-4
          --max-tokens 6144 --train 2000 --eval 64]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import re
import os
import gc
import time
import hashlib
from pathlib import Path

import torch

from .memory import cuda_allocated_bytes

from ..view import TOOLS as VIEW_TOOLS, combine_site, listing_instructions, part_site, plan as view_plan, \
    reject_digest_part, reply_prefix, view_note, view_site

INSTRUCTIONS = re.compile(r"Instructions:\n([\s\S]*?)\n\n(?:In eval|Eval also|$)")


# Canonical consumer-training defaults: a bare trajectory trainer selects the local-stage replay, same-slot channel
# target 0.1, source-state auxiliary gradient 0.05, control-row training, top-state transport, written child handoffs,
# depth 5, source-sized values, native writer and gold-boundary stop supervision, crisp SFT 1, KL 0.25, Muon and LoRA 16.
# These are training-policy defaults, not a foundation certificate or admission; explicit CLI settings override them and
# checkpoint identity rejects silent changes. Machine owners choose replay group size and CUDA envelopes independently.
# (Option names keep their historical `sketch_` spelling: they are part of every frozen checkpoint identity.)
CONSUMER_TRAINING_DEFAULTS = {
    "sketch_gradient": "local_stage",
    "sketch_target_weight": 0.1,
    "sketch_target_backbone_scale": 0.05,
    "train_control_rows": True,
    "content_transport": "top-state",
    "handover": "written",
    "write_curriculum": "joint",
    "write_depth": 5,
    "tokens_per_vector": 1.0,
    "writer_supervision": "native-value",
    "writer_length_policy": "native-value",
    "stop_supervision": "gold-native-boundary",
    "writer_text_weight": 1.0,
    "crisp_weight": 1.0,
    "distill": 0.25,
    "optimizer": "muon",
    "rank": 16,
    "lr": 1e-4,
    "lora_lr": 2e-5,
    "heads_lr": 3e-5,
    "max_tokens": 65536,
    "steps": 2200,
    "batch": 1,
    "backward_policy": "auto",
    "checkpoint_layers": True,
    "ffn_chunk_tokens": 1024,
    "token_cache_mib": 32,
    "eval_every": 64,
}


def apply_consumer_defaults(parser):
    actions = {action.dest: action for action in parser._actions}
    if not CONSUMER_TRAINING_DEFAULTS.keys() <= actions.keys():
        raise ValueError("canonical consumer defaults contain unknown trainer options")
    for name, value in CONSUMER_TRAINING_DEFAULTS.items():
        choices = actions[name].choices
        if choices is not None and value not in choices:
            raise ValueError("invalid canonical consumer default: " + name)
    parser.set_defaults(**CONSUMER_TRAINING_DEFAULTS)


VIEW_CHUNK_MARGIN = 32  # tokens: a decoded chunk can retokenize slightly longer than its window


def view_write_plan(tokenizer, value: str, window: int, tokens_per_vector: float, bound: int):
    """(chunks, per-chunk write lengths, combining write length) of a written view in training.

    The port's block bound caps every write. With a declared tokens-per-vector R > 0 a write is teacher-forced to
    ceil(tokens / R) vectors, so a chunk holds at most bound * R tokens (less a retokenization margin) and the view
    window shrinks to that; a value whose write would exceed the bound is chunked at the view site. Lengths are then
    min(bound, ceil(tokens / R)); the combining write is sized like its parts together, capped at the bound.
    Without R (0) lengths are None: the stop head decides, and it cannot exceed the bound."""
    if bound < 1:
        raise ValueError('the view bound must be positive')
    if tokens_per_vector and tokens_per_vector > 0:
        window = max(1, min(window, int(bound * tokens_per_vector) - VIEW_CHUNK_MARGIN))
    chunks = view_plan(tokenizer, value, window).chunks
    if not tokens_per_vector or tokens_per_vector <= 0:
        return chunks, [None] * len(chunks), None
    lengths = [min(bound, max(1, math.ceil(len(tokenizer(c, add_special_tokens=False)["input_ids"]) / tokens_per_vector)))
               for c in chunks]
    return chunks, lengths, min(bound, sum(lengths))


def crisp_messages(messages: list[dict], texts: dict[str, str], notes: dict[str, str], *,
                   neuralese_bodies: dict[str, str] | None = None) -> list[dict]:
    """Messages with every soft part as its text and every handover as its note: the crisp rendering."""
    return render(messages, lambda name: {"type": "text", "text": texts[name]}, notes,
                  neuralese_bodies=neuralese_bodies)


def render(messages: list[dict], soft_part, notes: dict[str, str], blocks: dict[str, str] | None = None,
           views: dict[str, str] | None = None,
           neuralese_bodies: dict[str, str] | None = None) -> list[dict]:
    """Converted messages → engine messages: `soft` parts via `soft_part(name)`; handover reads and writes as the
    written block where `blocks` has one (name → block ID), else as the crisp note; parts merged into text where no
    block remains."""
    blocks, views, neuralese_bodies = blocks or {}, views or {}, neuralese_bodies or {}
    out = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "soft":
                    parts.append(soft_part(part["name"]))
                elif part["type"] == "view":
                    # Written views show as the runtime lists them; otherwise the crisp cut-off preview.
                    if part["name"] in views:
                        parts.append({"type": "neuralese", "id": views[part["name"]]})
                        parts.append({"type": "text", "text": part.get("note") or view_note(part["holder"])})
                    else:
                        parts.append({"type": "text", "text": part["preview"]})
                elif part["type"] == "digest":
                    reject_digest_part(part)
                elif part["type"] == "read":
                    name = part["name"]
                    parts.append({"type": "neuralese", "id": blocks[name]} if name in blocks else
                                 {"type": "text", "text": part.get("source", notes.get(name, ""))})
                elif part["type"] == "neuralese":
                    # Ordinary text training can expand only a separately
                    # attested creation body. Other native blocks stay opaque.
                    body = neuralese_bodies.get(part.get("id"))
                    if body is None:
                        raise ValueError("unattested neuralese message body")
                    parts.append({"type": "text", "text": body})
                else:
                    parts.append({"type": "text", "text": part["text"]})
            message["content"] = ("".join(p["text"] for p in parts) if all(p["type"] == "text" for p in parts) else parts)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                args = call["function"]["arguments"]
                if isinstance(args, list):
                    rendered_args = []
                    for part in args:
                        if not isinstance(part, dict):
                            raise ValueError("tool-call argument part is not an object")
                        if part.get("type") == "text" and isinstance(part.get("text"), str):
                            rendered_args.append(part)
                        elif part.get("type") == "read":
                            name, source = part.get("name"), part.get("source")
                            if not isinstance(name, str) or not isinstance(source, str):
                                raise ValueError("tool-call read part lacks authenticated name/source")
                            if name in blocks:
                                rendered_args.append({"type": "neuralese", "id": blocks[name]})
                            else:
                                rendered_args.append({"type": "text", "text": json.dumps(source, ensure_ascii=False)[1:-1]})
                        elif part.get("type") == "neuralese":
                            raise ValueError("unresolved Neuralese block in tool-call arguments")
                        else:
                            raise ValueError("unresolved non-text tool-call argument part")
                    if all(part.get("type") == "text" for part in rendered_args):
                        args = "".join(part["text"] for part in rendered_args)
                        call = {**call, "function": {**call["function"], "arguments": args}}
                    else:
                        args = rendered_args
                        call = {**call, "function": {**call["function"], "arguments": rendered_args}}
                if not isinstance(args, (str, list)):
                    raise ValueError("tool-call arguments are not a string or typed-part sequence")
                if isinstance(args, list) and any(
                        not isinstance(part, dict) or part.get("type") not in ("text", "neuralese")
                        or (part.get("type") == "text" and not isinstance(part.get("text"), str))
                        for part in args):
                    raise ValueError("tool-call argument parts are not rendered text/Neuralese")
                if 'neuralese_code' in call:
                    if not isinstance(args, str):
                        raise ValueError("inline instruction sidecar cannot be combined with typed argument parts")
                    from .inline_instructions import render_inline_instruction_arguments
                    rendered_args=render_inline_instruction_arguments(args,call['neuralese_code'],blocks)
                    if rendered_args is None:raise ValueError('invalid inline instruction code sidecar')
                    call={k:v for k,v in call.items() if k!='neuralese_code'}
                    call={**call,'function':{**call['function'],'arguments':rendered_args}}
                    calls.append(call)
                    continue
                if isinstance(args, str) and '"$write"' in args:
                    def crisp_value(value):
                        """Expand write leaves recursively, retaining opaque blocks at their exact value path."""
                        if isinstance(value, dict) and "$write" in value:
                            site = value["$write"]
                            name = site["name"]
                            if name in blocks:
                                return [{"type": "neuralese", "id": blocks[name],
                                         "value_type": "string" if site.get("type", "Neuralese<string>") == "Neuralese<string>" else "unknown"}]
                            source = site["source"]
                            json_body = site.get("source_encoding") == "json" or site.get("type") == "Neuralese<unknown>"
                            return json.loads(source) if json_body else source
                        if isinstance(value, dict):
                            return {key: crisp_value(item) for key, item in value.items()}
                        if isinstance(value, list):
                            return [crisp_value(item) for item in value]
                        return value

                    value = crisp_value(json.loads(args))
                    arguments = json.dumps(value, ensure_ascii=False)
                    call = {**call, "function": {**call["function"], "arguments": arguments}}
                calls.append(call)
            message["tool_calls"] = calls
        out.append(message)
    return out


def reads(record: dict) -> set[str]:
    names = set()
    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "read" and isinstance(value.get("name"), str):
                names.add(value["name"])
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(record.get("messages") or [])
    return names


def _write_markers(value, path=()):
    """Yield ``(exact JSON value path, write descriptor)`` leaves without walking descriptor metadata."""
    if isinstance(value, dict) and '$write' in value:
        site=value['$write']
        if isinstance(site,dict) and isinstance(site.get('name'),str) and isinstance(site.get('source'),str):
            yield path,site
        return
    if isinstance(value,dict):
        for key,item in value.items():yield from _write_markers(item,path+(key,))
    elif isinstance(value,list):
        for index,item in enumerate(value):yield from _write_markers(item,path+(index,))


def write_sites(record: dict) -> list[tuple[str, dict, str, str]]:
    """Every actual target writer, including nested JSON values and inline bodies.

    The public tuple keeps its legacy top-level argument field. ``write_value_path`` supplies the exact nested
    path, and rejects duplicate IDs rather than guessing which occurrence owns a block.
    """
    sites=[]
    for call in (record.get('target') or {}).get('tool_calls') or []:
        arguments=json.loads(call['function']['arguments'])
        before={}
        for key,value in arguments.items():
            for _,write in _write_markers(value):
                sites.append((call['function']['name'],dict(before),key,write['name']))
            before[key]=value
        if 'neuralese_code' in call:
            from .inline_instructions import validate_inline_instruction_code
            checked=validate_inline_instruction_code(call['function']['arguments'],call['neuralese_code'])
            if not checked.valid:raise ValueError('invalid inline instruction code sidecar: '+str(checked.reason))
            before={}
            for key,value in arguments.items():
                if key=='code':break
                before[key]=value
            for write in checked.value.writes:
                sites.append((call['function']['name'],dict(before),'code',write.name))
    names=[site[3] for site in sites]
    if len(names)!=len(set(names)):
        raise ValueError('duplicate writer name in target')
    return sites


def write_value_path(record: dict, name: str | None = None) -> tuple[str | int, ...]:
    """Exact argument-tree path for one uniquely named writer in the target."""
    selected=name or record.get('_active_write_name')
    found=[]
    for call in (record.get('target') or {}).get('tool_calls') or []:
        arguments=json.loads(call['function']['arguments'])
        for key,value in arguments.items():
            found.extend(((key,)+path,write) for path,write in _write_markers(value)
                         if selected is None or write['name']==selected)
    if len(found)!=1:
        if not found and selected is not None and inline_write_prefix(record,selected) is not None:
            return ()
        raise ValueError('writer name is absent or has multiple value paths')
    return found[0][0]


def write_site_arguments(record: dict, name: str | None = None) -> dict:
    """Complete native call arguments for a writer, used when its value is nested."""
    selected=name or record.get('_active_write_name')
    matches=[]
    for call in (record.get('target') or {}).get('tool_calls') or []:
        arguments=json.loads(call['function']['arguments'])
        for value in arguments.values():
            matches.extend(write for _,write in _write_markers(value)
                           if selected is None or write['name']==selected)
        if 'neuralese_code' in call:
            from .inline_instructions import validate_inline_instruction_code
            checked=validate_inline_instruction_code(call['function']['arguments'],call['neuralese_code'])
            if not checked.valid:raise ValueError('invalid inline instruction code sidecar: '+str(checked.reason))
            matches.extend({'name':write.name} for write in checked.value.writes
                           if selected is None or write.name==selected)
        if matches:
            if len(matches)!=1:raise ValueError('writer name is ambiguous in target')
            return arguments
    raise ValueError('writer arguments are absent')


def write_site(record: dict) -> tuple[str, dict, str, str] | None:
    selected=record.get('_active_write_name')
    return next((site for site in write_sites(record) if selected is None or site[3]==selected),None)


def inline_write_prefix(record: dict, name: str) -> str | None:
    from .inline_instructions import validate_inline_instruction_code
    for call in (record.get('target') or {}).get('tool_calls') or []:
        if 'neuralese_code' not in call:continue
        checked=validate_inline_instruction_code(call['function']['arguments'],call['neuralese_code'])
        if not checked.valid:raise ValueError('invalid inline instruction code sidecar: '+str(checked.reason))
        pair=checked.value.prefix_and_body(name)
        if pair is not None:return pair[0]
    return None


def authenticated_capture_context_augmentation(record: dict, writer_name: str) -> tuple[dict | None, str | None]:
    """Expose authenticated full captures only as an explicit, separately hashable training augmentation."""
    from .inline_instructions import validate_inline_instruction_code
    selected = []
    for call in (record.get('target') or {}).get('tool_calls') or []:
        sidecar = call.get('neuralese_code')
        if not isinstance(sidecar, dict):
            continue
        checked = validate_inline_instruction_code(call.get('function', {}).get('arguments'), sidecar)
        if not checked.valid:
            raise ValueError('invalid inline instruction code sidecar: ' + str(checked.reason))
        for site in sidecar.get('sites', []):
            if not isinstance(site, dict) or site.get('name') != writer_name:
                continue
            plan = site.get('plan')
            binding = plan.get('capture_binding_plan') if isinstance(plan, dict) else None
            if not isinstance(binding, dict):
                continue
            for capture in binding.get('captures', []):
                if not isinstance(capture, dict) or capture.get('provider_full_value_visible') is not False:
                    continue
                snapshot = capture.get('host_snapshot')
                preview = capture.get('displayed_preview')
                augmentation = capture.get('context_augmentation')
                if (not isinstance(snapshot, dict) or not isinstance(preview, dict) or
                        not isinstance(augmentation, dict) or
                        augmentation.get('mode') != 'authenticated-runtime-capture-snapshot' or
                        augmentation.get('value_sha256') != snapshot.get('value_sha256') or
                        snapshot.get('value') != capture.get('value')):
                    raise ValueError('truncated capture lacks authenticated context-augmentation provenance')
                selected.append({'name': capture['name'], 'type': capture['type'], 'source': capture['source'],
                                 'full_value': capture['value'], 'value_sha256': snapshot['value_sha256'],
                                 'displayed_preview': preview['text'],
                                 'preview_sha256': preview['sha256'],
                                 'omitted_characters': preview['omitted_characters']})
    if not selected:
        return None, None
    content = ('Explicit training context augmentation for a captured code writer. The original provider-visible '
               'conversation showed only each shortened preview below; the omitted text was not fully visible there. '
               'The full value is supplied here from the same invocation\'s authenticated host capture snapshot.\n' +
               '\n'.join(f"Capture {item['name']} ({item['type']}, source={item['source']}): "
                         f"historical_preview={json.dumps(item['displayed_preview'], ensure_ascii=False)}; "
                         f"omitted_characters={item['omitted_characters']}; "
                         f"full_authenticated_value={json.dumps(item['full_value'], ensure_ascii=False)}; "
                         f"snapshot_sha256={item['value_sha256']}" for item in selected))
    digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
    content += f'\ncontext_augmentation_sha256={digest}'
    return {'role': 'user', 'content': content}, digest


def write_value_type(record: dict) -> str:
    site=write_site(record)
    if site is None:return 'string'
    name=site[3]
    if inline_write_prefix(record,name) is not None:return 'string'
    for call in (record.get('target') or {}).get('tool_calls') or []:
        for _,write in _write_markers(json.loads(call['function']['arguments'])):
            if write['name']==name:
                kind=write.get('type','Neuralese<string>')
                if kind=='Neuralese<string>':return 'string'
                if kind=='Neuralese<unknown>' or write.get('source_encoding')=='json':return 'unknown'
                raise ValueError('nested writer has unsupported declared value type')
    raise ValueError('selected writer contract is missing')


def target_writes(record: dict) -> set[str]:
    return {site[3] for site in write_sites(record)}


def target_write(record: dict) -> str | None:
    site=write_site(record)
    return site[3] if site else None


def handover_notes(record: dict) -> dict[str, str]:
    """Note texts by name: from the record's compaction calls and from its reads (which carry their note)."""
    notes = {}
    def remember(name,source):
        if name in notes and notes[name]!=source:
            raise ValueError('one write name has conflicting source values')
        notes[name]=source
    def remember_reads(value):
        if isinstance(value, dict):
            if value.get("type") == "read" and "source" in value:
                remember(value["name"], value["source"])
            for child in value.values(): remember_reads(child)
        elif isinstance(value, list):
            for child in value: remember_reads(child)
    remember_reads(record.get("messages", []))
    for message in record["messages"] + ([record["target"]] if record.get("target") else []):
        for call in message.get("tool_calls") or []:
            if 'neuralese_code' in call:
                from .inline_instructions import validate_inline_instruction_code
                checked=validate_inline_instruction_code(call['function']['arguments'],call['neuralese_code'])
                if not checked.valid:raise ValueError('invalid inline instruction code sidecar: '+str(checked.reason))
                for write in checked.value.writes:remember(write.name,write.source)
            if '"$write"' in call["function"]["arguments"]:
                for _,write in _write_markers(json.loads(call["function"]["arguments"])):
                    remember(write['name'],write['source'])
    return notes


def authenticated_recurrence_context_view(records, pieces):
    """Resolve captured context parts against admitted writers or exact context receipts.

    Writer-backed parts become `read` parts so the ordinary written-value path
    retains their differentiable producer edge. Attested context-only bodies
    become uniquely named reads without a producer. Unresolved rows are held
    with source identity and a concrete reason.
    """
    import copy
    from ..data.text_corpus import (
        _attested_neuralese_message_bodies, _attested_provider_expanded_reads,
        _canonical, _sha, _soft_writer_sources, authenticated_crisp_context_messages,
    )
    rows = list(records)
    source_hashes = {r.get("id", ""): (r.get("_source_record_sha256") or
                     _sha(_canonical(dict(r)).encode("utf-8"))) for r in rows}
    writer_sources = _soft_writer_sources(rows, source_hashes)
    producers = {}
    for row in rows:
        for name in target_writes(row):
            if name in producers:
                raise ValueError("duplicate recurrence writer: " + name)
            producers[name] = row
    prepared, excluded, decisions = [], [], []

    def walk(value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for source in rows:
        record = copy.deepcopy(source)
        parts = [p for p in walk(record.get("messages") or [])
                 if p.get("type") in ("read", "neuralese")]
        if not parts:
            prepared.append(record)
            continue
        rid = source.get("id")
        row_decisions = []
        try:
            split = source.get("split")
            groups = sorted(set(g for g in (source.get("source_groups") or [])
                                if isinstance(g, str) and g))
            if split not in ("train", "test") or not groups:
                raise ValueError("context row lacks eligible split/source groups")
            # This shared renderer authenticates every neuralese body, provider
            # read, capture augmentation, and nested tool-argument reference.
            authenticated_crisp_context_messages(source, pieces, writer_sources)
            provider_reads = _attested_provider_expanded_reads(
                source, writer_sources, split=split, source_groups=groups)
            bodies, body_attestations = _attested_neuralese_message_bodies(
                source, writer_sources, split=split, source_groups=groups,
                authenticated_context_bodies={a["block_id"]: a for a in provider_reads
                                              if isinstance(a.get("body"), str)})
            body_by_id = {a["block_id"]: a for a in body_attestations}
            provider_by_name = {}
            for item in provider_reads:
                name = item.get("write_name")
                if isinstance(name, str): provider_by_name.setdefault(name, []).append(item)
                block_id = item.get("block_id")
                if isinstance(block_id, str):
                    provider_by_name.setdefault("soft-state:" + block_id, []).append(item)
            inline_site = ((source.get("source_ref") or {}).get("inline_instruction_site") or {})
            inline_site_data = inline_site.get("site") if isinstance(inline_site, dict) else None
            inline_site_valid = (isinstance(inline_site_data, dict)
                and (inline_site.get("validation") or {}).get("valid") is True
                and not (inline_site.get("validation") or {}).get("reasons")
                and isinstance(inline_site_data.get("template_segments"), list)
                and all(isinstance(x, str) for x in inline_site_data["template_segments"]))
            inline_site_text = ("".join(inline_site_data["template_segments"])
                                if inline_site_valid else None)

            def replace_parts(value):
                if isinstance(value, dict):
                    if value.get("type") == "neuralese":
                        block_id = value.get("id")
                        att = body_by_id.get(block_id)
                        wrapped = bodies.get(block_id)
                        if att is None or not isinstance(wrapped, str) or not wrapped.startswith("<|neuralese|>") or not wrapped.endswith("<|/neuralese|>"):
                            row_decisions.append({"kind": "neuralese", "block_id": block_id,
                                "status": "held", "body_sha256": None,
                                "reason": "neuralese context lacks an exact body attestation"})
                            raise ValueError("neuralese context lacks an exact body attestation")
                        body = wrapped[len("<|neuralese|>"):-len("<|/neuralese|>")]
                        writer_name = att.get("write_name")
                        local_writer = att.get("source_kind") == "approved_writer_target_source" and writer_name in producers
                        name = writer_name if local_writer else "context-only:" + block_id
                        row_decisions.append({"kind": "neuralese", "block_id": block_id,
                            "source_sha256": att.get("body_sha256"), "source_kind": att.get("source_kind"),
                            "writer_record_id": att.get("writer_record_id"), "writer_name": writer_name,
                            "gradient_edge": bool(local_writer)})
                        return {"type": "read", "name": name, "source": body}
                    if value.get("type") == "read":
                        name, body = value.get("name"), value.get("source")
                        source_name = name
                        if not isinstance(name, str) or not isinstance(body, str):
                            row_decisions.append({"kind": "read", "name": name if isinstance(name, str) else None,
                                "status": "held", "body_sha256": None,
                                "reason": "read context lacks exact name/source"})
                            raise ValueError("read context lacks exact name/source")
                        if name in producers:
                            expected = handover_notes(producers[name]).get(name)
                            if expected is None or expected != body:
                                row_decisions.append({"kind": "read", "name": name, "status": "held",
                                    "body_sha256": _sha(body.encode("utf-8")),
                                    "reason": "read body differs from exact selected writer source"})
                                raise ValueError("read body differs from exact selected writer source")
                            evidence = {"source_kind": "exact-target-writer-source",
                                        "writer_record_id": producers[name].get("id"),
                                        "body_sha256": _sha(body.encode("utf-8"))}
                            edge = True
                        else:
                            candidates = [a for a in provider_by_name.get(name, [])
                                          if a.get("body") == body and a.get("body_sha256") == _sha(body.encode("utf-8"))]
                            if (not candidates and name.startswith("inline-site:")
                                    and inline_site_valid and inline_site_text == body):
                                row_decisions.append({"kind": "read", "name": name,
                                    "source_kind": "validated-inline-site-template-source",
                                    "status": "resolved-context-only",
                                    "body_sha256": _sha(body.encode("utf-8")),
                                    "gradient_edge": False})
                                return {**value, "name": "context-only:inline-site:" + name.partition(":")[2]}
                            if not candidates:
                                row_decisions.append({"kind": "read", "name": name, "status": "held",
                                    "body_sha256": _sha(body.encode("utf-8")),
                                    "reason": "read body has no exact writer or provider receipt"})
                                raise ValueError("read body has no exact writer or provider receipt")
                            evidence = {"source_kind": candidates[0].get("source_kind"),
                                        "writer_record_id": candidates[0].get("writer_record_id"),
                                        "body_sha256": candidates[0].get("body_sha256"),
                                        "block_id": candidates[0].get("block_id")}
                            edge = candidates[0].get("writer_target_selected") is True and name in producers
                            name = name if edge else "context-only:" + (candidates[0].get("block_id") or _sha(name.encode())[:24])
                        row_decisions.append({"kind": "read", "name": name, "source_name": source_name, **evidence,
                                              "gradient_edge": bool(edge)})
                        return {**value, "name": name}
                    return {key: replace_parts(child) for key, child in value.items()}
                if isinstance(value, list):
                    return [replace_parts(child) for child in value]
                return value
            record["messages"] = replace_parts(record.get("messages") or [])
            prepared.append(record)
            decisions.append({"id": rid, "source_record_sha256": source_hashes.get(rid),
                              "split": split, "source_groups": groups, "status": "eligible",
                              "references": row_decisions})
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            excluded.append({"id": rid, "source_record_sha256": source_hashes.get(rid),
                             "split": source.get("split"),
                             "source_groups": sorted(set(g for g in (source.get("source_groups") or [])
                                                         if isinstance(g, str) and g)),
                             "reason": str(exc)[:300], "references": row_decisions})
            decisions.append({"id": rid, "source_record_sha256": source_hashes.get(rid),
                              "split": source.get("split"), "status": "held", "reason": str(exc)[:300],
                              "references": row_decisions})
    summary = {"schema": "natlang.recurrence-context-review/1",
               "source_records": len(rows), "prepared_records": len(prepared),
               "excluded_context_records": len(excluded),
               "prepared_by_split": {split: sum(r.get("split") == split for r in prepared)
                                     for split in ("train", "test")},
               "excluded_by_split": {split: sum(r.get("split") == split for r in excluded)
                                     for split in ("train", "test")},
               "decisions": decisions, "excluded": excluded}
    return prepared, summary


def native_writer_prefix(record, apply_template, *, value_type=None):
    """Actual template reply through the chosen argument and any preceding eval code."""
    from ..serve.chat import write_reply,write_value_text
    call,before,argument,name=write_site(record)
    value_type=value_type or write_value_type(record)
    path=write_value_path(record,name)
    arguments=write_site_arguments(record,name) if len(path)>1 else before
    path_arg=path if len(path)>1 else None
    prefix=write_reply(apply_template,call,arguments,argument,value_type,argument_path=path_arg)[0]
    code_prefix=inline_write_prefix(record,name)
    if code_prefix is not None:
        prefix+=write_value_text(apply_template,call,before,argument,code_prefix,value_type)
    return prefix


def producer_text_target(record, texts, names):
    """Gold ordinary reply at a producer, preserving its incoming soft values.

    Its own output must stay gold text here, so body-token supervision is not
    replaced by the generated opaque payload that the reader objective consumes.
    """
    own = target_writes(record)
    ancestors = {name: block for name, block in names.items() if name not in own}
    return render([record['target']], lambda name: {'type': 'text', 'text': texts[name]},
                  handover_notes(record), ancestors)[0]




def _progress(records, label: str):
    """Yield records, logging each one's wall time to stderr: evaluation passes run before the first update and
    otherwise print nothing until done, so a slow backbone looks like a hang."""
    import sys
    import time
    start = time.time()
    for index, record in enumerate(records):
        began = time.time()
        yield record
        print(json.dumps({"event": "eval_progress", "label": label, "record": index + 1, "of": len(records),
                          "seconds": round(time.time() - began, 2), "elapsed": round(time.time() - start, 1)}),
              file=sys.stderr, flush=True)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", required=True)
    parser.add_argument("--pieces", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--member-weight", type=float, default=0.0,
                        help="nested-family students (MAPLE_NESTED §4a): each update also trains one member (in rotation) "
                             "on the crisp rendering of its last record, whole trajectory: CE + KL(full || member), times "
                             "this weight; members' private parts become trainable. 0 disables (members still evaluated)")
    parser.add_argument("--member-tokens", type=int, default=2048, help="members train and evaluate on the last N tokens")
    parser.add_argument("--member-eval", type=int, default=4, help="held records for the per-member evaluation")
    parser.add_argument("--member-mask-system", action=argparse.BooleanOptionalAction, default=True,
                        help="member term and member evaluation leave out the system prompt and tool definitions: they "
                             "are identical across records, so members memorized them (2026-10-09: ~89%% of member "
                             "windows were 16-grams of other records) and the evaluation could not tell")
    parser.add_argument("--member-full-weight", type=float, default=0.0,
                        help="with the member term, the full model's own crisp CE on the same window (the shared "
                             "weights the member term moves stay anchored to the full model's crisp trajectory)")
    parser.add_argument("--qat-latent-lr", type=float, default=0.0,
                        help="Maple QAT dense latents get their own AdamW groups at this rate times each matrix's "
                             "ternary scale (codes flip after moving ~0.5 of it); 0: Muon at the backbone rate, under "
                             "which codes practically never flip")
    parser.add_argument("--crisp-weight", type=float, default=0.0, help="additional ordinary-text SFT, backward separately before the same optimizer step; preserves interpreter policy alongside soft-return learning")
    parser.add_argument("--projection-anchor-weight", type=float, default=1.0,
                        help="gold-aligned auxiliary relative-MSE anchor for full content and shallow feedback projections against detached raw next-token embeddings; set 0 only for an explicit diagnostic ablation")
    parser.add_argument("--projection-anchor-decay-steps", type=int, default=0,
                        help="the projection anchor's weight falls linearly to 0 over this many updates from the start "
                             "of the lineage stage (owner 2026-10-09: the Neuralese-to-token-embedding anchor belongs to "
                             "the warm-in only; afterwards readers adapt via --read-adapter). 0: constant")
    parser.add_argument("--read-adapter", action=argparse.BooleanOptionalAction, default=False,
                        help="reader-side Neuralese input adaptation: every vector entering the read port passes a "
                             "zero-initialised full-rank residual map trained by the readers (heads.NeuraleseReadAdapter)")
    parser.add_argument("--projection-anchor-backbone-scale", type=float, default=0.05,
                        help="gradient multiplier from the projection anchor into backbone states; projection parameters receive full gradient")
    parser.add_argument("--writer-text-weight", type=float, default=None, help="teacher-forced gold producer reply under its actual soft/ancestor context; additional local writer objective")
    parser.add_argument("--stop-supervision", choices=["generated-length", "gold-native-boundary"], default="generated-length", help="teach stop on coherent gold value states with balanced terminal/continue loss")
    parser.add_argument("--writer-supervision", choices=["full-reply", "native-value"], default="native-value", help="teacher-force the gold body under the exact forced writer prefix; full-reply reproduces earlier supervision")
    parser.add_argument("--writer-length-policy", choices=["source-text", "native-value"], default="native-value", help="supervised producer length uses its exact native template value, not source JSON")
    parser.add_argument("--content-transport", choices=["learned-residual", "raw-identity", "top-state"], default="learned-residual", help="explicit raw identity warm-up or learned content residual")
    parser.add_argument("--content-residual-initialization", choices=["preserve", "fresh-zero"], default="preserve",
                        help="explicit raw-to-learned transition: zero previously bypassed residual and only its optimizer slots")
    parser.add_argument("--curriculum-change", action="append", default=[], choices=["tokens_per_vector", "writer_text_weight", "write_depth", "write_curriculum", "max_writes", "max_write_vectors", "content_transport", "content_residual_initialization", "writer_length_policy", "writer_supervision", "stop_supervision", "steps", "sketch_gradient", "member_weight", "member_tokens", "member_eval", "member_mask_system", "member_full_weight", "qat_latent_lr", "projection_anchor_weight", "projection_anchor_backbone_scale", "projection_anchor_decay_steps", "read_adapter"], help="explicitly permit named curriculum changes at --continue-from while preserving optimizer/RNG and fixed data")
    parser.add_argument('--max-write-vectors', type=int, default=None,
                        help='explicit port payload bound, distinct from prompt context; constant-stop capacity can extend without changing weights/moments')
    parser.add_argument("--continue-from", help="explicit new code stage preserving full optimizer/RNG; requires identical data and training controls")
    parser.add_argument("--soft-init", help="warm-start matching soft parameters from a prior soft-params or full recurrence checkpoint; new pieces are text-initialized")
    parser.add_argument("--heads", default=None, help="port heads checkpoint (soft parameters are read through them)")
    parser.add_argument("--base", default=None)
    parser.add_argument("--bank", default=None, help="system-prompt bank to initialise current pieces from")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=None,
                        help="records/chains per optimizer step (default: 1 for sampled-chain, 4 for joint)")
    parser.add_argument("--lr", type=float, default=1e-3, help="soft parameters")
    parser.add_argument("--rank", type=int, default=0, help="LoRA rank (0: soft parameters only)")
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5,
                        help="learning rate for native full-layer backbone training")
    parser.add_argument("--quantization", default=None,
                        help="the recipe quantization component for this stage (JSON; train/quantization.py): one "
                             "precision point per update, drawn by weight among BF16 and the active points; a gate "
                             "column per deploy precision at each evaluation. Absent: BF16 only")
    parser.add_argument("--backbone-training", choices=["auto", "full", "lora", "qat", "latent"], default="auto",
                        help="auto selects native full-layer training or Maple QAT; lora is an explicit diagnostic policy")
    parser.add_argument("--max-tokens", type=int, default=6144, help="skip records whose crisp prompt is longer")
    parser.add_argument("--train", type=int, default=2000, help="training records to read")
    parser.add_argument("--eval", type=int, default=64, help="held-out records")
    parser.add_argument("--handover", choices=["crisp", "written"], default="crisp",
                        help="handoffs (handover notes, child calls' results): their crisp text, or written by their producer")
    parser.add_argument('--write-curriculum', choices=['joint', 'sampled-chain'], default='joint',
                        help='joint uses all handoffs unless max-writes is set; sampled-chain picks one handoff per nested level each step')
    parser.add_argument("--max-writes", type=int, default=0,
                        help="write at most this many of a record's handoffs per step, chosen at random each step; the rest "
                             "read their crisp text (bounds memory: each written value prefills its producer with gradient; 0: all)")
    parser.add_argument("--write-depth", type=int, default=2,
                        help="levels of written values inside written values' producers (1: producers read crisp text)")
    parser.add_argument("--only-handover", action="store_true", help="only records that read or write a note")
    parser.add_argument("--view", choices=["preview", "written"], default="preview",
                        help="view parts: the crisp preview, or the view the model writes at view's template write site")
    parser.add_argument("--view-window", type=int, default=4096,
                        help="value tokens per view write site in training (longer values are viewed in chunks)")
    parser.add_argument("--view-tokens-per-vector", type=float, default=None,
                        help="teacher-forced length of written views in training: ceil(tokens of each write's own text / "
                             "this), every write within the port's block bound (values chunked to fit; "
                             "view_write_plan); 0: the stop head decides; default: --tokens-per-vector")
    parser.add_argument("--purpose-contrast", type=float, default=0.0,
                        help="view stage: weight of the purpose contrast on records with view_stage.contrast_instructions "
                             "(a second view of the same value written for a partner purpose must serve this record's "
                             "reader worse by --purpose-margin nats/token; hinge; requires --view written; 0: off)")
    parser.add_argument("--purpose-margin", type=float, default=0.1, help="margin of the purpose contrast (nats/token)")
    parser.add_argument("--stop-pg", type=float, default=0.0,
                        help="train the stop head on written values by policy gradient with this length cost per vector (0: off)")
    parser.add_argument("--tokens-per-vector", type=float, default=0.0,
                        help="size each written value from the crisp text it stands for (the note's text, the view's listing "
                             "preview): ceil(tokens / this) vectors, the stop head trained on that boundary (0: the stop head decides)")
    parser.add_argument("--sketch-gradient", choices=["unroll", "one_step", "local_stage"], default="unroll")
    parser.add_argument("--local-stage-batch-size", type=int, default=1, help="isolated sketch stages per tensor batch;1 is sequential reference; explicit memory/performance control")
    parser.add_argument("--sketch-target-backbone-scale", type=float, default=0.05, help="auxiliary sketch-target input gradient multiplier; projection receives full gradient")
    parser.add_argument("--sketch-target-weight", type=float, default=0.)
    parser.add_argument("--train-control-rows", action=argparse.BooleanOptionalAction, help="train/save/restore LM control rows for close-token stopping")
    parser.add_argument("--stop-weight", type=float, default=1.0, help="weight of the stop-boundary loss on source-sized writes")
    parser.add_argument("--heads-lr", type=float, default=1e-4, help="the writer's port heads, when notes or views are written")
    parser.add_argument("--detach-write-context", action="store_true",
                        help="no gradient into the write sites' context (saves memory; soft prompts there then do not learn from writing)")
    parser.add_argument("--distill", type=float, default=1.0, help="weight of the self-distillation term on written notes")
    parser.add_argument("--context-weight", type=float, default=1.0, help="CE weight of each record's new prompt text (instructions, inputs, tool results since the last assistant reply); the whole trajectory is a target, not only replies (owner, 2026-10-07)")
    parser.add_argument("--feedback-weight", type=float, default=0.25, help="relative weight, inside --context-weight, of tool results and other mechanical feedback (non-assistant turns after the first reply)")
    parser.add_argument("--context-coverage", choices=["last-reply", "records"], default="last-reply",
                        help="which earlier prompt text other records supervise (weight 0 in the context term): "
                             "last-reply assumes every earlier reply has its own record; records zeroes only what an "
                             "earlier training record of the same trajectory (exact message-prefix chain) covers, so "
                             "corpora that keep some turns per trajectory (harness_bench) supervise the whole trajectory")
    parser.add_argument("--cohort-weights", default=None,
                        help="JSON fractions per record cohort (record field 'cohort', default native), naming exactly "
                             "the loaded training cohorts and summing to one; draws cohort, then document (record). "
                             "Naming a cohort admits nothing")
    parser.add_argument("--qualification-cohort", default="native",
                        help="the cohort whose held records drive the evaluations and probes; other cohorts' held "
                             "records are reported separately in cohort_strata")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-gb", type=float, default=None,
                        help="CUDA envelope (default: ledger NATLANG_CUDA_MEMORY_GB, else 90%% of free CUDA memory)")
    parser.add_argument('--backward-policy', choices=['joint', 'staged', 'auto'], default='joint')
    parser.add_argument('--graph-memory-gb', type=float, default=0,
                        help='auto joint graph high-water budget; 0 derives from the CUDA envelope')
    parser.add_argument('--graph-headroom-gb', type=float, default=.35,
                        help='reserve within CUDA envelope for backward temporaries and optimizer state')
    parser.add_argument("--activation-offload-gb", type=float, default=0,
                        help="CPU budget for saved activations; exact gradients, no recomputation or detached writes")
    parser.add_argument('--checkpoint-layers', action=argparse.BooleanOptionalAction,
                        help='recompute layer activations during backward to reduce memory; preserves full recurrence gradients')
    parser.add_argument('--checkpoint-elide-rng', action='store_true',
                        help='skip RNG snapshot/restore inside verified deterministic native layer checkpoints')
    parser.add_argument('--producer-batch-size', type=int, default=1,
                        help='maximum tensor batch of independent staged raw/native producer calls; memory admission can choose fewer')
    parser.add_argument('--joint-producer-batching', action='store_true',
                        help='also batch independent raw/native writers while retaining the complete joint graph')
    parser.add_argument('--producer-batch-memory-gb', type=float, default=0,
                        help='optional budget for one released staged frontier, separate from the complete joint graph budget')
    parser.add_argument('--checkpoint-attention-only', action='store_true',
                        help='opt-in resource policy: checkpoint attention, retain convolution activations; requires --checkpoint-layers and more graph memory')
    parser.add_argument('--staged-checkpoint-attention-only', action='store_true',
                        help='opt-in: retain convolution activations only for staged writer replays whose geometry fits the graph budget; preserve full reader/joint checkpoints')
    parser.add_argument('--ffn-chunk-tokens', type=int, default=0,
                        help='token-local FFN chunks reduce transient allocations without context truncation')
    parser.add_argument('--token-cache-mib', type=int, default=0,
                        help='bounded CPU token-ID cache for this fixed tokenizer; 0 disables it')
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw')
    from .optim_restore import add_optimizer_restore_arguments
    add_optimizer_restore_arguments(parser)
    parser.add_argument('--checkpoint-every', type=int, default=25,
                        help='full-state points: every N updates, a multiple of --eval-every when that is set (the end '
                             'and stops also write the full state)')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='evaluation points: held-out soft and written-vs-shuffled probes every N updates; 0: none '
                             'between the initial reference and the final report (the stage end, its gate)')
    apply_consumer_defaults(parser)
    parser.add_argument("--inspect-training-config", action="store_true", help="print effective defaults and overrides without loading models or starting training")
    args = parser.parse_args(argv)
    option_defaults = {action.dest: action.default for action in parser._actions if action.dest != 'help'}
    anchor_now = [args.projection_anchor_weight]  # the scheduled projection-anchor weight of the current update
    if args.inspect_training_config:
        print(json.dumps(vars(args), sort_keys=True, indent=2))
        return
    if args.token_cache_mib < 0:
        raise ValueError('negative token cache budget')
    cohort_weights = None
    if args.cohort_weights is not None:
        from .recipe import validate_cohort_weights
        cohort_weights = json.loads(args.cohort_weights)
        validate_cohort_weights(cohort_weights)
        if args.qualification_cohort not in cohort_weights:
            raise ValueError('the qualification cohort must be one of --cohort-weights')
    if not math.isfinite(args.context_weight) or args.context_weight < 0:
        raise ValueError('context weight must be finite and nonnegative')
    if not math.isfinite(args.feedback_weight) or not 0 <= args.feedback_weight <= 1:
        raise ValueError('feedback weight must be finite and between zero and one')
    if args.local_stage_batch_size < 1 or (args.local_stage_batch_size != 1 and args.sketch_gradient != 'local_stage'):
        raise ValueError('local stage batches require local_stage and a positive size')
    if args.producer_batch_size < 1:
        raise ValueError('producer batch size must be positive')
    if args.joint_producer_batching and args.producer_batch_size < 2:
        raise ValueError('joint producer batching requires producer batch size >=2')
    if not math.isfinite(args.producer_batch_memory_gb) or args.producer_batch_memory_gb < 0:
        raise ValueError('invalid producer batch memory budget')
    if args.producer_batch_size > 1 and (args.tokens_per_vector != 1 or args.content_transport != 'raw-identity'
            or args.writer_supervision != 'native-value' or args.stop_supervision != 'gold-native-boundary'
            or args.stop_pg or args.max_writes or args.write_curriculum != 'joint'):
        raise ValueError('producer batching requires deterministic raw/native gold-boundary full-DAG supervision')
    from .staging import StagedWrites, resolve_values, GraphBudgetExceeded, graph_memory_budget
    from .recurrence import curriculum_max_writes
    args.max_writes = curriculum_max_writes(args.write_curriculum, args.max_writes)
    if args.batch is None:
        args.batch = 1 if args.write_curriculum == 'sampled-chain' else 4
    if args.steps < 1 or args.batch < 1 or args.checkpoint_every < 1 or args.write_depth < 1 or args.activation_offload_gb < 0 or args.ffn_chunk_tokens < 0 or args.eval_every < 0:
        raise ValueError('invalid recurrence training controls')
    if (args.checkpoint_attention_only or args.staged_checkpoint_attention_only) and not args.checkpoint_layers:
        raise ValueError('attention-only checkpointing requires --checkpoint-layers')
    if args.checkpoint_attention_only and args.staged_checkpoint_attention_only:
        raise ValueError('choose global or staged-writer attention-only checkpointing')
    if args.max_write_vectors is not None and args.max_write_vectors < 1:
        raise ValueError('invalid write capacity')
    if not math.isfinite(args.tokens_per_vector) or args.tokens_per_vector < 0:
        raise ValueError('invalid tokens per vector')
    if args.curriculum_change and not args.continue_from:
        raise ValueError('curriculum changes require explicit continuation checkpoint')
    if args.stop_supervision == 'gold-native-boundary' and (args.writer_supervision != 'native-value' or args.tokens_per_vector != 1):
        raise ValueError('gold native stop supervision requires native value supervision and one token per vector')
    if args.writer_supervision == 'native-value' and args.writer_length_policy != 'native-value':
        raise ValueError('native-value writer supervision requires native-value length policy')
    if args.writer_text_weight is not None and (not math.isfinite(args.writer_text_weight) or args.writer_text_weight < 0):
        raise ValueError('invalid producer text supervision weight')
    if not math.isfinite(args.crisp_weight) or args.crisp_weight < 0:
        raise ValueError('invalid crisp SFT weight')
    if not math.isfinite(args.projection_anchor_weight) or args.projection_anchor_weight < 0:
        raise ValueError('invalid projection anchor weight')
    if not math.isfinite(args.projection_anchor_backbone_scale) or not 0.0 <= args.projection_anchor_backbone_scale <= 1.0:
        raise ValueError('projection anchor backbone scale must be finite and between zero and one')
    if args.graph_memory_gb < 0 or args.graph_headroom_gb <= 0:
        raise ValueError('invalid graph memory budget')
    if args.backward_policy == 'auto' and (args.stop_pg or args.view == 'written'):
        # Staged backward needs deterministic handoffs and preview views; with written views or the stop policy the
        # auto policy is the joint graph (its memory envelope still applies).
        print(json.dumps({'backward_policy': 'joint', 'declared': 'auto',
                          'reason': 'written views or stop policy: staging needs deterministic handoffs and previews'}), flush=True)
        args.backward_policy = 'joint'
    if args.backward_policy != 'joint' and (args.stop_pg or args.view == 'written'):
        raise ValueError('staging currently requires deterministic handoffs and preview views')
    if args.purpose_contrast and (args.view != 'written' or args.purpose_contrast < 0 or
                                  not math.isfinite(args.purpose_contrast) or not math.isfinite(args.purpose_margin)):
        raise ValueError('the purpose contrast needs --view written and a finite nonnegative weight and margin')
    if args.device.startswith('cuda'):
        free, total = torch.cuda.mem_get_info()
        args.memory_gb = args.memory_gb or float(os.environ.get('NATLANG_CUDA_MEMORY_GB') or free / 2**30 * .9)
        envelope = min(args.memory_gb, total / 2**30)
        graph_budget = (args.graph_memory_gb or envelope - args.graph_headroom_gb) * 2**30
        if graph_budget <= 0 or graph_budget >= envelope * 2**30:
            raise ValueError('graph budget must leave backward headroom within the CUDA envelope')
        if args.producer_batch_memory_gb and args.producer_batch_memory_gb > envelope - args.graph_headroom_gb:
            raise ValueError('producer batch budget exceeds the envelope with backward headroom')
    else:
        graph_budget = 0
        args.memory_gb = args.memory_gb or 8
    batch_graph_budget = args.producer_batch_memory_gb * 2**30 or graph_budget

    from ..prompt_bank import load_bank, save_bank
    from ..serve import load_engine
    from ..serve.chat import RequestError, call_reply, write_reply, render_messages
    from ..serve.grad import GradSession, encode_text
    from .execution import Prefilled, write_generated
    from .losses import stop_boundary_loss
    from ..serve.store import make_block
    from .adapters import lora_state

    out = Path(args.out)
    checkpoint_path = out / 'checkpoint.pt'
    if out.exists() and any(out.iterdir()) and not checkpoint_path.exists():
        raise ValueError('existing recurrence output has no resumable checkpoint; use a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    from .trajectory_state import atomic_checkpoint, trajectory_optimizer, validate_resume, validate_continuation, gradient_norm, clip_finite_gradients
    def digest_file(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        return digest.hexdigest()
    identity = {'options': {k: v for k, v in vars(args).items() if k not in {'inspect_training_config', 'out', 'memory_gb', 'activation_offload_gb', 'checkpoint_every', 'checkpoint_minutes', 'eval_minutes', 'backward_policy', 'graph_memory_gb', 'graph_headroom_gb', 'continue_from', 'curriculum_change', 'optimizer_state', 'optimizer_added', 'checkpoint_attention_only', 'staged_checkpoint_attention_only', 'checkpoint_elide_rng', 'producer_batch_size', 'producer_batch_memory_gb', 'token_cache_mib', 'joint_producer_batching', 'local_stage_batch_size'} and not (k == 'writer_text_weight' and v is None)},
                'files': {str(Path(p).resolve()): digest_file(p) for p in [args.records, args.pieces, args.heads, args.bank, args.soft_init] if p},
                'code': {str(p.resolve()): digest_file(p) for p in Path(__file__).resolve().parents[1].rglob('*.py')}}
    if args.continue_from:
        identity['continuation'] = {'checkpoint_sha256': digest_file(args.continue_from),
                                    'path': str(Path(args.continue_from).resolve()),
                                    'curriculum_changes': args.curriculum_change}
    from ..common.artifact_paths import artifact_refs
    # Each input bound to its role once, at launch, for every loader of the checkpoints written below.
    input_roles = artifact_refs(vars(args), identity['files'], ('base', 'heads', 'records', 'pieces', 'bank', 'soft_init'))
    resumed = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path.exists() else None
    new_continuation = resumed is None and bool(args.continue_from)
    if resumed is not None:
        changed_code = validate_resume(resumed, identity, defaults=option_defaults)
        if changed_code:
            print(json.dumps({'event': 'resumed_on_new_code', 'changed_modules': len(changed_code),
                              'modules': [Path(m).name for m in changed_code][:20]}), flush=True)
    elif args.continue_from:
        resumed = torch.load(args.continue_from, map_location='cpu', weights_only=False)
        validate_continuation(resumed, identity, allowed_changes=args.curriculum_change, defaults=option_defaults)
    if args.content_residual_initialization == 'fresh-zero':
        if not args.continue_from or args.content_transport != 'learned-residual':
            raise ValueError('fresh-zero residual requires an explicit learned-residual continuation')
        if new_continuation and (resumed.get('port_config', {}).get('content_transport') != 'raw-identity'
                or 'content_residual_initialization' not in args.curriculum_change):
            raise ValueError('fresh-zero transition must explicitly declare a previously bypassed raw residual')
    if args.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2**30 / total))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = load_engine(args.base, heads_checkpoint=args.heads, device=args.device)
    if args.token_cache_mib:
        from ..serve.token_cache import TokenCache
        engine._token_cache = TokenCache(engine.tokenizer, args.token_cache_mib * 2**20)
    from ..model.capacity import set_write_capacity, source_vector_length
    capacity = args.max_write_vectors
    if capacity is None:
        capacity = (resumed or {}).get('port_config', {}).get('max_length', engine.heads.max_length)
    set_write_capacity(engine.heads, capacity)
    engine.heads.set_content_transport(args.content_transport)
    engine.max_block = capacity
    if args.stop_supervision == 'gold-native-boundary' and engine.heads.read_markers:
        raise ValueError('gold native stop supervision requires raw-token-v1')
    for p in engine.backbone.parameters():
        p.requires_grad_(False)
    if not 0 <= args.sketch_target_backbone_scale <= 1:
        raise ValueError("sketch target backbone gradient scale must be between 0 and 1")
    if args.sketch_target_weight < 0 or (args.sketch_target_weight and args.sketch_gradient not in ('one_step', 'local_stage')):
        raise ValueError('positive sketch target weight requires one_step or local_stage')
    if engine.heads.autoregressive and (args.sketch_gradient not in ('one_step', 'local_stage') or args.sketch_target_weight <= 0):
        raise ValueError('latent-sketch-v2 training requires one_step or local_stage and a positive sketch target weight')
    if engine.heads.autoregressive and args.stop_weight and not args.train_control_rows:
        raise ValueError('close-token stop training requires explicit --train-control-rows')
    session = GradSession(engine)

    texts, piece_kinds = {}, {}
    with open(args.pieces) as stream:
        for line in stream:
            piece = json.loads(line)
            texts[piece["name"]] = piece["text"]
            piece_kinds[piece["name"]] = piece.get("kind")
    with open(args.records) as stream:
        source_records = [json.loads(line) for line in stream]
    # Records converted before the view rename are refused before any model work, naming their conversion.
    for record in source_records:
        for message in record.get("messages") or []:
            for part in message.get("content") if isinstance(message.get("content"), list) else []:
                reject_digest_part(part)
    if "prompt:digest" in texts:
        reject_digest_part({"type": "digest"})
    records, context_review = authenticated_recurrence_context_view(source_records, texts)
    context_review_path = out / "recurrence-context-review.json"
    context_review_path.write_text(json.dumps(context_review, ensure_ascii=False, indent=2) + "\n")
    bank = load_bank(args.bank) if args.bank else None
    params, leaf_ids, from_bank = {}, {}, []

    producers = {}
    if args.handover == "written":
        # Every record whose target writes a note, by note name: the producer of that note's block.
        for record in records:
            for name in target_writes(record):
                if name in producers:raise ValueError('duplicate producer for '+name)
                producers[name]={**record,'_active_write_name':name}

    backbone, heads = engine.backbone, engine.heads
    backbone.ffn_chunk_tokens = args.ffn_chunk_tokens
    backbone.checkpoint_layers = args.checkpoint_layers
    backbone.checkpoint_attention_only = args.checkpoint_attention_only
    active_staging = [None]
    selective_writer_replays = [0]
    from .memory_estimator import (AdaptiveGraphMemory, backbone_memory_layout,
                                   geometry_bytes, producer_geometry_bytes,
                                   writer_batch_kind)
    shared_kv_prefix = bool(args.checkpoint_layers and backbone.fast and
                            getattr(backbone, 'attention_checkpoint_prefixes', False))
    geometry_version = 'shared-prefix-v1' if shared_kv_prefix else 'full-prefix-v1'
    native_gold_tape = args.writer_supervision == 'native-value' and bool(
        args.writer_text_weight or (args.stop_supervision == 'gold-native-boundary' and args.stop_weight))
    if native_gold_tape:
        geometry_version += ':native-gold-tape-v1'
    plain_layers = (sum(not backbone.is_attention(i) for i in range(backbone.num_layers))
                    if args.checkpoint_attention_only else 0)
    if plain_layers:
        geometry_version += ':attention-only-v1'
    # A new gradient replay changes retained tape and transient branch geometry.
    # Never inherit earlier joint-fit routes across that resource boundary.
    geometry_version += (f':writer-{heads.profile}:{args.sketch_gradient}:k{heads.cutoff}'
                         f':local-group{args.local_stage_batch_size if args.sketch_gradient == "local_stage" else 1}')
    if args.sketch_gradient == 'local_stage':
        geometry_version += ':local-stage-geometry-v1'
    memory_estimator = AdaptiveGraphMemory(resumed.get('memory_estimator') if resumed else None,
                                           geometry_version=geometry_version)
    memory_layout = backbone_memory_layout(
        backbone, checkpointed=args.checkpoint_layers,
        shared_kv_prefix=shared_kv_prefix, uncheckpointed_layers=plain_layers,
        stage_group_size=args.local_stage_batch_size if args.sketch_gradient == "local_stage" else 0,
        sketch_cutoff=heads.cutoff)
    geometry_cache = {}
    from .recurrence import ProducerMemo, is_acyclic, independent_frontier, dependency_frontiers
    dependencies = {name: (reads(record) | set(handover_notes(record))) - target_writes(record)
                    for name, record in producers.items()}
    share_producers = is_acyclic(dependencies) and not args.max_writes and not args.stop_pg

    def placeholder(name: str) -> str:
        """A block ID that stands for a value written afresh each step; its rows always come from the leaves."""
        digest = hashlib.sha256(name.encode()).digest()
        return "nz1_" + base64.b32encode(digest).decode().rstrip("=").lower()

    def source_length(text: str | None) -> int | None:
        """The supervised length of a write standing for `text`, or None (the stop head decides)."""
        if not args.tokens_per_vector or not text:
            return None
        return source_vector_length(len(engine._tokens(text)), args.tokens_per_vector, heads.max_length)

    native_sources = {}
    def producer_source(name, producer):
        source = handover_notes(producer).get(name)
        if args.writer_length_policy == 'source-text' or source is None:
            return source
        if name not in native_sources:
            from ..serve.chat import write_value_text
            tool, before, argument, own = write_site(producer)
            if own != name:
                raise ValueError('producer source does not match its native write site')
            value_type = write_value_type(producer)
            value = json.loads(source) if value_type == 'unknown' else source
            path=write_value_path(producer,name)
            arguments=write_site_arguments(producer,name) if len(path)>1 else before
            native_sources[name] = write_value_text(
                lambda m, g: engine.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=g),
                tool, arguments, argument, value, value_type,
                argument_path=path if len(path)>1 else None)
        return native_sources[name]

    if args.tokens_per_vector:
        # Fail configuration before any update, rather than discover a clipped
        # producer only when a later reader reaches it.
        for name, producer in producers.items():
            source_length(producer_source(name, producer))

    def write(messages, tools, prefix, leaves, source: str | None = None, resource_choice=None, gold_stop_supervised=False,
              length: int | None = None):
        previous = backbone.checkpoint_attention_only
        try:
            return write_impl(messages, tools, prefix, leaves, source, resource_choice, gold_stop_supervised, length)
        finally:
            backbone.checkpoint_attention_only = previous

    def write_impl(messages, tools, prefix, leaves, source: str | None = None, resource_choice=None, gold_stop_supervised=False,
                   length: int | None = None):
        """The write procedure with gradients (S3 `unroll_write`): the site's prompt (soft parts from `leaves`), the
        forced prefix and the open marker, then the sketch recurrence until the stop head stops. The stop decisions
        are detached; the payload carries gradients into the writer (feedback, content projection, LoRA) and into
        every soft parameter of the site."""
        prompt = render_messages(messages, tools, engine._template, engine.specials)
        items = session._items(prompt.segments, prompt.blocks, prompt.escape_nonce) + [("tok", t) for t in engine._tokens(prefix)]
        context = session._embed_items(items, leaves)
        write_context_lengths.append(context.shape[1])
        if args.detach_write_context:
            context = context.detach()
        if (args.staged_checkpoint_attention_only and active_staging[0] is not None
                and torch.is_grad_enabled() and args.device.startswith('cuda')
                and getattr(backbone, 'attention_checkpoint_prefixes', False)):
            from .memory_estimator import selective_writer_fits
            choose = lambda: selective_writer_fits(
                context.shape[1], source_length(source) or heads.max_length, memory_layout,
                baseline=cuda_allocated_bytes(), budget=graph_budget,
                plain_layers=sum(not backbone.is_attention(i) for i in range(backbone.num_layers)))
            backbone.checkpoint_attention_only = resource_choice.resolve(choose) if resource_choice else choose()
            selective_writer_replays[0] += int(backbone.checkpoint_attention_only)
        try:
            from .execution import prefill_write_context
            pre = prefill_write_context(backbone, heads, context)
        except torch.OutOfMemoryError:
            print(json.dumps({'status': 'producer_out_of_memory', 'context_tokens': context.shape[1],
                              'ffn_chunk_tokens': args.ffn_chunk_tokens}), flush=True)
            raise
        target = length if length is not None else source_length(source)
        if target is not None:
            # Sized from the crisp text it stands for: no stop decision; the stop head learns the boundary.
            written = write_generated(backbone, heads, pre, args.sketch_gradient, sketch_target_backbone_scale=args.sketch_target_backbone_scale, local_stage_batch_size=args.local_stage_batch_size, length=target)
            if args.stop_weight and not gold_stop_supervised and torch.is_grad_enabled():
                boundary_terms.append(args.stop_weight * stop_boundary_loss(written))
        else:
            written = write_generated(backbone, heads, pre, args.sketch_gradient, sketch_target_backbone_scale=args.sketch_target_backbone_scale, local_stage_batch_size=args.local_stage_batch_size, sample=bool(args.stop_pg), generator=stop_generator)
        if args.sketch_target_weight and torch.is_grad_enabled():
            if written.sketch_target_loss is None:
                raise ValueError('sketch self-target requires one_step or local_stage writes')
            boundary_terms.append(args.sketch_target_weight * written.sketch_target_loss)
        # Gold lengths are already known on the host. Reading them back from
        # CUDA adds a completion wait without adding any information.
        n = target if target is not None else int(written.lengths[0])
        lengths.append(n)
        if target is None and args.stop_pg and torch.is_grad_enabled():
            # Log-probability of the sampled stop decisions under the stop head (continue after 1..n-1, stop after n
            # unless the write ran to the maximum), differentiable in the stop head.
            logits = written.stop_logits[0].float()
            logp = torch.nn.functional.logsigmoid(-logits[:n - 1]).sum()
            if not bool(written.truncated[0]):
                logp = logp + torch.nn.functional.logsigmoid(logits[n - 1])
            stop_terms.append((logp, n))
        return written.payload[0, :n]

    lengths: list[int] = []
    projection_anchor_values: list[float] = []
    contrast_values: list[float] = []  # purpose contrast: CE(partner-purpose view) - CE(own view), this update
    write_context_lengths: list[int] = []
    writer_batches: list[int] = []
    stop_terms: list = []  # (log-probability of the stop decisions, length) of this record's writes
    boundary_terms: list = []  # stop-boundary losses of this record's source-sized writes
    stop_generator = torch.Generator().manual_seed(args.seed)
    write_choice = random.Random(args.seed)
    baseline = {"value": None}

    prefixes = {}

    def site_prefix(record):
        """The forced reply before the written argument, in the model's own rendering of the producer's call."""
        call, before, argument, _ = write_site(record)
        value_type = write_value_type(record) if not engine.heads.read_markers else "string"
        own=write_site(record)[3]
        code_prefix=inline_write_prefix(record,own)
        path=write_value_path(record,own)
        template_arguments=write_site_arguments(record,own) if len(path)>1 else before
        key = (call, json.dumps(template_arguments, sort_keys=True), argument, path, value_type, code_prefix)
        if key not in prefixes:
            prefixes[key]=native_writer_prefix(record,
                lambda m,g:engine.tokenizer.apply_chat_template(m,tokenize=False,add_generation_prompt=g),value_type=value_type)
        return prefixes[key]

    def prepare_note(name, leaves, depth, visiting, memo):
        producer = producers[name]
        names, payloads = {}, {}
        if depth < args.write_depth:
            # Depth counts writes, once per edge. Previously incrementing here
            # AND in written_values silently made depth3 only two write layers.
            names, payloads = written_values(producer, leaves, depth, visiting + (name,), memo)
        messages = render(producer["messages"], lambda n: {"type": "neuralese", "id": leaf_ids[n]}, handover_notes(producer),
                          names, names)
        capture_context, _capture_context_sha256 = authenticated_capture_context_augmentation(producer, name)
        if capture_context is not None:
            messages = [*messages, capture_context]
        from .memory_estimator import ReplayResourceChoice
        resource_choice = ReplayResourceChoice()
        def replay():
            begin = len(boundary_terms)
            result = write(messages, producer.get("tools"), site_prefix(producer),
                           resolve_values({**leaves, **payloads}), source=producer_source(name, producer),
                           resource_choice=resource_choice, gold_stop_supervised=args.stop_supervision == 'gold-native-boundary')
            terms = boundary_terms[begin:]
            del boundary_terms[begin:]
            return result, terms
        def gold_replay():
            if producer.get('split') != 'train' or producer.get('training_admission', {}).get('approved') is not True:
                raise ValueError('producer gold supervision requires admitted training split')
            if args.writer_supervision == 'native-value':
                loss = session.supervised_continuation_loss(
                    messages, producer.get('tools'), site_prefix(producer), producer_source(name, producer),
                    resolve_values({**leaves, **payloads}), text_weight=args.writer_text_weight,
                    stop_weight=args.stop_weight if args.stop_supervision == 'gold-native-boundary' else 0.,
                    projection_anchor_weight=anchor_now[0],
                    projection_anchor_backbone_scale=args.projection_anchor_backbone_scale)
                if anchor_now[0]:
                    projection_anchor_values.append(float(session.last_projection_anchor_loss))
                return loss
            target = producer_text_target(producer, texts, names)
            loss = session.supervised_text_loss(
                {'messages': messages, 'tools': producer.get('tools'), 'target': target},
                resolve_values({**leaves, **payloads}),
                # Keep the anchor's configured scale independent of the
                # writer CE multiplier applied to the combined return below.
                projection_anchor_weight=(anchor_now[0] / args.writer_text_weight
                                          if args.writer_text_weight else 0.),
                projection_anchor_backbone_scale=args.projection_anchor_backbone_scale)
            if anchor_now[0]:
                projection_anchor_values.append(float(session.last_projection_anchor_loss))
            return args.writer_text_weight * loss
        auxiliary = gold_replay if (args.writer_text_weight or (args.stop_supervision == 'gold-native-boundary' and args.stop_weight)) and torch.is_grad_enabled() else None
        return producer, messages, payloads, replay, auxiliary

    def note_payload(name, leaves, depth=1, visiting=(), memo=None):
        """One producer and its recursively prepared dependencies, sharing the same writer surface."""
        if memo is None:
            memo = ProducerMemo(share_producers)
        def compute():
            _, _, _, replay, auxiliary = prepare_note(name, leaves, depth, visiting, memo)
            if active_staging[0] is not None:
                return active_staging[0].add(replay, auxiliary=auxiliary)
            value, terms = replay()
            if auxiliary is not None:
                terms = [sum(terms) + auxiliary()]
            boundary_terms.extend(terms)
            return value
        return memo.write(name, depth, compute)

    def producer_frontier(chosen, leaves, depth, visiting, memo):
        """Prepare dependencies first, then admit pinned independent tensor groups."""
        if not independent_frontier(chosen, dependencies):
            raise ValueError('dependent producers cannot share a tensor batch')
        jobs = []
        for name in chosen:
            if (name, depth) in memo.values:
                continue
            producer, messages, payloads, replay, auxiliary = prepare_note(name, leaves, depth, visiting, memo)
            prompt = render_messages(messages, producer.get('tools'), engine._template, engine.specials)
            items = session._items(prompt.segments, prompt.blocks, prompt.escape_nonce)
            items += [('tok', t) for t in engine._tokens(site_prefix(producer))]
            scope = {**leaves, **payloads}
            resolved = resolve_values(scope)
            width = sum(1 if kind == 'tok' else resolved[value].shape[0] if value in resolved
                        else engine.lookup(value).length for kind, value in items)
            jobs.append({'name': name, 'items': items, 'scope': scope, 'width': width,
                         'length': source_length(producer_source(name, producer)),
                         'replay': replay, 'auxiliary': auxiliary})
        while jobs:
            count = min(args.producer_batch_size, len(jobs))
            # Joint execution retains preceding tapes. Staged execution releases
            # them and can use its separately declared frontier envelope.
            frontier_budget = batch_graph_budget if active_staging[0] is not None else graph_budget
            baseline_bytes = cuda_allocated_bytes() if args.device.startswith('cuda') else 0
            while count > 1:
                group = tuple(jobs[:count])
                width, vectors = max(j['width'] for j in group), max(j['length'] for j in group)
                raw = count * geometry_bytes(width, vectors, **{**memory_layout, 'uncheckpointed_layers': 0})
                predicted = memory_estimator.predict(writer_batch_kind(count), width, vectors, raw)
                if not frontier_budget or baseline_bytes + predicted <= frontier_budget:
                    break
                count -= 1
            group = tuple(jobs[:count])
            del jobs[:count]
            if count == 1:
                job = group[0]
                if active_staging[0] is not None:
                    node = active_staging[0].add(job['replay'], auxiliary=job['auxiliary'])
                else:
                    node, terms = job['replay']()
                    if job['auxiliary'] is not None:
                        terms = [sum(terms) + job['auxiliary']()]
                    boundary_terms.extend(terms)
                memo.write(job['name'], depth, lambda node=node: node)
                continue
            width, vectors = max(j['width'] for j in group), max(j['length'] for j in group)
            plain = sum(not backbone.is_attention(i) for i in range(backbone.num_layers))
            selective_raw = count * geometry_bytes(width, vectors, **{**memory_layout, 'uncheckpointed_layers': plain})
            selective_kind = writer_batch_kind(count, selective=True)
            selective_estimate = memory_estimator.predict(selective_kind, width, vectors, selective_raw)
            selective = bool(active_staging[0] is not None and args.staged_checkpoint_attention_only and args.checkpoint_layers and
                             (count <= 2 or len(memory_estimator.calibration(selective_kind, width, vectors)) >= 3) and
                             (not frontier_budget or baseline_bytes + selective_estimate <= frontier_budget))
            raw = selective_raw if selective else count * geometry_bytes(width, vectors, **{**memory_layout, 'uncheckpointed_layers': 0})
            kind = writer_batch_kind(count, selective=selective)
            def compute_batch(group=group, selective=selective, frontier_budget=frontier_budget):
                # Membership and checkpoint policy are captured once.
                # Rebuild embedded scope on replay, retaining all child VJPs.
                previous = backbone.checkpoint_attention_only
                backbone.checkpoint_attention_only = selective
                selective_writer_replays[0] += len(group) if selective else 0
                try:
                    contexts = [session._embed_items(j['items'], resolve_values(j['scope']))[0] for j in group]
                    if any(c.shape[0] != j['width'] for c, j in zip(contexts, group)):
                        raise RuntimeError('producer batch scope geometry changed during replay')
                    if args.detach_write_context:
                        contexts = [c.detach() for c in contexts]
                    from .execution import prefill_write_contexts
                    with graph_memory_budget(frontier_budget if args.device.startswith('cuda') else 0):
                        pre = prefill_write_contexts(backbone, heads, contexts)
                        # Validate/max the known lengths on CPU; unroll_write
                        # transfers them once for its returned row metadata.
                        sizes = torch.tensor([j['length'] for j in group])
                        written = write_generated(backbone, heads, pre, args.sketch_gradient, sketch_target_backbone_scale=args.sketch_target_backbone_scale, local_stage_batch_size=args.local_stage_batch_size, lengths=sizes)
                    write_context_lengths.extend(j['width'] for j in group)
                    lengths.extend(j['length'] for j in group)
                    writer_batches.append(len(group))
                    penalties = [[] for _ in group]
                    if args.sketch_target_weight and torch.is_grad_enabled():
                        row_losses = written.sketch_target_loss_by_row
                        if row_losses is None or row_losses.shape != (len(group),):
                            raise RuntimeError('batched sketch self-target did not return one loss per producer row')
                        penalties = [[args.sketch_target_weight * row_losses[row]] for row in range(len(group))]
                    return [written.payload[row, :j['length']] for row, j in enumerate(group)], penalties
                finally:
                    backbone.checkpoint_attention_only = previous
            def observe_batch(values, retained, kind=kind, width=width, vectors=vectors, raw=raw):
                memory_estimator.observe(kind, width, vectors, raw, retained)
            if active_staging[0] is not None:
                nodes = active_staging[0].add_batch(compute_batch, auxiliaries=[j['auxiliary'] for j in group],
                                                  observe=observe_batch)
            else:
                before = retained_tape_bytes() if args.device.startswith('cuda') else 0
                nodes, penalty_rows = compute_batch()
                if args.device.startswith('cuda'):
                    observe_batch(nodes, max(0, retained_tape_bytes() - before))
                for job, terms in zip(group, penalty_rows):
                    if job['auxiliary'] is not None:
                        terms = [sum(terms) + job['auxiliary']()]
                    boundary_terms.extend(terms)
            for job, node in zip(group, nodes):
                memo.write(job['name'], depth, lambda node=node: node)
        return {name: memo.values[(name, depth)] for name in chosen}

    view_prefix = []
    view_tokens_per_vector = args.tokens_per_vector if args.view_tokens_per_vector is None else args.view_tokens_per_vector
    if not math.isfinite(view_tokens_per_vector) or view_tokens_per_vector < 0:
        raise ValueError('view tokens per vector must be finite and nonnegative')
    view_lengths: list[int] = []  # vectors of each view write (all within the port's block bound)

    def view_payload(record, part, leaves):
        """The view of a value by its plan at view's template write site, every write differentiable. Lengths
        (view_write_plan): a write never exceeds the port's block bound; a value whose write would is chunked at
        the site so that each chunk's write fits; with --view-tokens-per-vector R a write's teacher-forced length is
        ceil(its own text's tokens / R) (the stop head learns that boundary), else the stop head decides."""
        if not view_prefix:
            view_prefix.append(reply_prefix(
                lambda m, g: engine.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=g)))
        system = [{"type": "neuralese", "id": leaf_ids["prompt:view"]}]
        written = {}

        def site_write(messages, length):
            payload = write(messages, VIEW_TOOLS, view_prefix[0], {**leaves, **written}, length=length)
            block = placeholder(f"{part['name']}#{len(written)}")
            written[block] = payload
            return block

        # A view part may carry the instructions it is written for (an agent's intent at the tool call whose output
        # it views, HARNESS_BENCH.md); a faithful part (view-stage reconstruction, data/view_records.py) is written
        # without instructions, `view(x)`; otherwise it is written for the receiving call, as the runtime's listing does.
        instructions = part.get("instructions")
        if part.get("faithful"):
            instructions = None
        elif not instructions:
            crisp = crisp_messages(record["messages"], texts, handover_notes(record))
            opening = next((m["content"] for m in crisp if m["role"] == "user" and isinstance(m["content"], str)), "")
            found = INSTRUCTIONS.search(opening)
            instructions = listing_instructions(part["holder"], part["value_type"], found.group(1) if found else "")
        chunks, lengths, combine = view_write_plan(engine.tokenizer, part["source"], args.view_window,
                                                   view_tokens_per_vector, heads.max_length)
        if len(chunks) == 1:
            block = site_write(view_site(system, chunks[0], instructions), lengths[0])
        else:
            parts = [site_write(part_site(system, chunk, instructions, i, len(chunks)), n)
                     for i, (chunk, n) in enumerate(zip(chunks, lengths))]
            # The combining write: sized like its parts together (at most the bound) when sized, except under the
            # stop policy (--stop-pg), where its length is the stop head's choice at the length cost.
            block = site_write(combine_site(system, instructions, parts), None if args.stop_pg else combine)
        view_lengths.extend(int(written[b].shape[0]) for b in written)
        return written[block]

    def written_values(record, leaves, depth=0, visiting=(), memo=None, reader_only=False):
        """Blocks written for this record this step: handoffs it reads or shows (notes, child results), views in its
        listing. Inside a producer (depth > 0) its own target's value is not one of them. Returns (name → placeholder
        ID, placeholder ID → payload)."""
        names, payloads = {}, {}
        if memo is None:
            memo = ProducerMemo(share_producers)
        if args.handover == "written":
            own = target_writes(record) if depth or reader_only else set()
            visible = reads(record) if reader_only else reads(record) | set(handover_notes(record))
            chosen = [name for name in sorted(visible)
                      if name in producers and name not in own and name not in visiting]
            if args.max_writes and len(chosen) > args.max_writes:
                chosen = sorted(write_choice.sample(chosen, args.max_writes))
            frontier = None
            if (args.producer_batch_size > 1 and share_producers and len(chosen) > 1
                    and (active_staging[0] is not None or (args.joint_producer_batching and torch.is_grad_enabled()))):
                frontier = {}
                for ready in dependency_frontiers(chosen, dependencies):
                    frontier.update(producer_frontier(ready, leaves, depth + 1, visiting, memo))
            for name in chosen:
                names[name] = placeholder(name)
                payloads[names[name]] = frontier[name] if frontier is not None else note_payload(name, leaves, depth + 1, visiting, memo)
        if args.view == "written":
            for message in record["messages"]:
                for part in message.get("content") if isinstance(message.get("content"), list) else []:
                    reject_digest_part(part)
                    if part["type"] == "view":
                        names[part["name"]] = placeholder(part["name"])
                        payloads[names[part["name"]]] = view_payload(record, part, leaves)
        return names, payloads

    def soft_messages(record, names):
        return render(record["messages"], lambda name: {"type": "neuralese", "id": leaf_ids[name]}, handover_notes(record),
                      names, names)

    def target_of(record, names):
        return render([record["target"]], lambda name: {"type": "text", "text": texts[name]}, handover_notes(record), names)[0]

    def prompt_tokens(record):
        crisp = crisp_messages(record["messages"], texts, handover_notes(record))
        rendered = render_messages(crisp, record.get("tools"), engine._template, engine.specials)
        return len(session._items(rendered.segments, rendered.blocks, rendered.escape_nonce))

    def native_neuralese_prompt(record):
        return any(isinstance(m.get("content"), list) and any(part.get("type") == "neuralese" for part in m["content"])
                   for m in record["messages"])

    # Records whose prompt already holds native Neuralese blocks have no attested crisp body to render here;
    # they are omitted (and counted), never silently expanded.
    # Cohorts (record field 'cohort', default native): without --cohort-weights every record must be the
    # qualification cohort and the selection below is unchanged; with it, --train caps each cohort's training records,
    # held records of the qualification cohort drive the evaluations and other cohorts' held records are a stratum.
    from .record_cohorts import record_cohort, RecordCohortSampler, trajectory_coverage
    train, held_pool, skipped = [], [], {"long": 0, "no-target": 0, "native-neuralese-prompt": 0,
                                        "unresolved-authenticated-context": context_review["excluded_context_records"]}
    cohort_held, train_per_cohort = {}, {}
    for record in records:
            cohort = record_cohort(record)
            if cohort_weights is None and cohort != args.qualification_cohort:
                raise ValueError('records of cohort ' + cohort + ' need explicit --cohort-weights')
            if not record.get("target"):
                skipped["no-target"] += 1
                continue
            if args.only_handover and not (reads(record) or handover_notes(record)):
                continue
            bucket = held_pool if record.get("split") == "test" else train
            if bucket is held_pool and cohort != args.qualification_cohort:
                bucket = cohort_held.setdefault(cohort, [])
            if bucket is train and train_per_cohort.get(cohort, 0) >= args.train:
                continue
            if native_neuralese_prompt(record):
                skipped["native-neuralese-prompt"] += 1
                continue
            if prompt_tokens(record) > args.max_tokens:
                skipped["long"] += 1
                continue
            bucket.append(record)
            if bucket is train:
                train_per_cohort[cohort] = train_per_cohort.get(cohort, 0) + 1
    if cohort_weights is not None and set(train_per_cohort) != set(cohort_weights):
        raise ValueError('--cohort-weights must name exactly the loaded training cohorts: ' +
                         ', '.join(sorted(train_per_cohort)))
    sampler = RecordCohortSampler(train, cohort_weights, seed=args.seed) if cohort_weights is not None else None
    coverage = trajectory_coverage(train) if args.context_coverage == 'records' else {}
    from .trajectory_probe import select_held, select_paired_held, source_groups
    probe_policy = 'typed-factual-reciprocal-readers-v1'
    held = select_held(held_pool, args.eval)
    paired_held, held_probe_accounting = select_paired_held(held_pool, args.eval, producers, piece_kinds)
    paired_train, train_probe_accounting = select_paired_held(train, args.eval, producers, piece_kinds)
    probe_selection = {'policy': probe_policy, 'ce_selected_ids': [r['id'] for r in held],
                       'held_pool_rows': len(held_pool), 'held_written': held_probe_accounting,
                       'train_written': train_probe_accounting,
                       'scope': 'Declared typed factual-disjoint reciprocal readers only; exclusions explicit. '
                                'Forced native lengths; not autonomous stop or broad task quality qualification.'}
    cohort_strata_held = {cohort: select_held(pool, args.eval) for cohort, pool in sorted(cohort_held.items())}
    if cohort_strata_held:
        probe_selection['cohort_strata_ids'] = {cohort: [r['id'] for r in chosen]
                                                for cohort, chosen in cohort_strata_held.items()}
    probe_selection_hash = hashlib.sha256(json.dumps(probe_selection, sort_keys=True).encode()).hexdigest()
    (out / 'eval-selection.json').write_text(json.dumps(probe_selection, indent=2) + '\n')
    if sampler is not None or coverage:
        covered = [n for n in coverage.values() if n]
        (out / 'cohort-sampling.json').write_text(json.dumps({
            'sampling': sampler.receipt() if sampler is not None else 'cursor over the selected records (one cohort)',
            'qualification_cohort': args.qualification_cohort, 'train_per_cohort': train_per_cohort,
            'held_strata': {cohort: len(chosen) for cohort, chosen in cohort_strata_held.items()},
            'context_coverage': args.context_coverage,
            'coverage': {'records': len(coverage), 'continuing_an_earlier_record': len(covered),
                         'covered_replies_total': sum(covered)} if coverage else None,
            'admission_granted': False}, indent=2) + '\n')
    # Soft parameters for the names the selected records use (a corpus has thousands of instructions texts).
    strata_records = [record for chosen in cohort_strata_held.values() for record in chosen]
    used_names = {part["name"] for record in train + held + strata_records + paired_held + paired_train + list(producers.values()) for message in record["messages"]
                  if isinstance(message.get("content"), list) for part in message["content"] if part["type"] == "soft"}
    if args.view == "written":
        used_names.add("prompt:view")
    from .trajectory_state import soft_initialization, resumed_initial_rows, iteration_rng_state, restore_iteration_rng
    warm_rows = soft_initialization(args.soft_init, texts, backbone.config.hidden_size, profile=heads.profile) if args.soft_init and resumed is None else {}
    saved_rows = resumed_initial_rows(resumed, used_names, backbone.config.hidden_size)
    from_previous = []
    for name in sorted(used_names):
        text = texts[name]
        piece = name.removeprefix("prompt:")
        if resumed is not None:
            # Original rows preserve block IDs; trained values and optimizer
            # state are restored below. Re-encoding here only wastes compute.
            rows = saved_rows[name]
        elif name in warm_rows:
            rows = warm_rows[name]
            from_previous.append(name)
        elif bank and name.startswith("prompt:") and piece in bank.rows:
            rows = bank.rows[piece]
            from_bank.append(name)
        else:
            rows = encode_text(engine, text).payload
        block = engine.store.put(make_block(rows, engine.dialect, type="Neuralese<SystemPrompt>",
                                            producer={"kind": "bank" if name in from_bank else "text-encode", "text": text}))
        params[name] = torch.nn.Parameter(block.payload.clone().float().to(engine.device))
        leaf_ids[name] = block.id
    init = {name: p.detach().clone() for name, p in params.items()}
    handovers = sum(1 for r in train if handover_notes(r))
    if args.handover == "written":
        # Training records that read or show a note come first in the mix's accounting; their producers may lie
        # outside the read window, which is why every producer was indexed.
        handovers = sum(1 for r in train if reads(r) or handover_notes(r))
    print(json.dumps({"train": len(train), "heldout": len(held), "skipped": skipped, "soft_params": len(params),
                      "from_bank": len(from_bank), "from_previous_soft": len(from_previous), "records_with_handover": handovers,
                      "handover": args.handover, "note_producers": len(producers),
                      **({"train_per_cohort": train_per_cohort,
                          "held_strata": {c: len(r) for c, r in cohort_strata_held.items()}} if sampler else {})}), flush=True)

    from .backbone_policy import configure_backbone_training, backbone_trainable_state, resolve_backbone_policy
    args.backbone_training=resolve_backbone_policy(engine.backbone,args.backbone_training)
    if args.backbone_training=='lora' and args.rank<1:
        raise ValueError('explicit LoRA policy requires --rank positive')
    backbone_named = configure_backbone_training(engine.backbone,args.backbone_training,rank=args.rank or 16)
    from .quantization import load_component
    from ..common.paths import root as path_root
    quant = load_component(args.quantization, root=path_root('repo'))
    preserve = behaviour = None
    if quant is not None:
        installed = quant.install(engine.backbone.hf, layer_count=engine.backbone.num_layers)
        print(json.dumps({'event': 'quantization_installed', 'groups': installed, 'stage': quant.stage}), flush=True)
        from .quantization import Behaviour, Preserve
        if quant.component.get('preserve') and quant.component['preserve'].get('weight', 0) > 0:
            preserve = Preserve(quant.component['preserve'], quant.stage['id'])
        if quant.component.get('gate'):
            behaviour = Behaviour(quant.component['gate'], engine.tokenizer, preserve.held if preserve else None,
                                  root=path_root('repo'))
    from ..maple.family import evaluate_members, family_members, member_backward, private_parameters, window_labels
    family = family_members(engine.backbone)
    codes = None
    if args.backbone_training == 'qat':
        # QAT code dynamics at each periodic evaluation (flips from the base, since the last one, oscillation).
        from ..maple.ternary import CodeTracker
        codes = CodeTracker(engine.backbone.hf)
    if args.member_weight and not family:
        raise ValueError('--member-weight needs a nested-family student')
    if args.member_weight:
        known = {id(p) for _, p in backbone_named}
        for name, p in private_parameters(engine.backbone):
            if id(p) not in known:
                p.requires_grad_(True)
                backbone_named.append((name, p))

    def member_eval_windows():
        """The held records of the member evaluation, pinned in the run directory at first use: a change of the held
        selection code (as at Maple recurrence step ~223) must not swap the evaluated records mid-run."""
        pinned = out / 'member-eval-ids.json'
        held_by_id = {r['id']: r for r in held}
        if pinned.exists():
            ids = [i for i in json.loads(pinned.read_text())['ids'] if i in held_by_id]
        else:
            ids = [r['id'] for r in held[:args.member_eval]]
            pinned.write_text(json.dumps({'ids': ids, 'mask_system': args.member_mask_system}, indent=1) + '\n')
        return [member_window(held_by_id[i]) for i in ids]

    def member_window(record):
        """The crisp rendering of a record (prompt, tool output and target: whole trajectory), last --member-tokens."""
        notes = handover_notes(record)
        messages = crisp_messages(record["messages"], texts, notes) + render(
            [record["target"]], lambda name: {"type": "text", "text": texts[name]}, notes)
        tools = record.get("tools") or None
        tokens = engine._tokens(engine.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False))
        context = 1
        if args.member_mask_system and messages and messages[0].get("role") == "system":
            # The system prompt and tool definitions stay context: identical across records, they taught members
            # nothing but memorization (and dominated the member evaluation).
            context = len(engine._tokens(engine.tokenizer.apply_chat_template(messages[:1], tools=tools, tokenize=False)))
        start = max(0, len(tokens) - args.member_tokens)
        ids = torch.tensor([tokens[start:]], device=args.device)
        return ids, window_labels(ids, min(max(1, context - start), ids.shape[1] - 1))
    lora = [parameter for _,parameter in backbone_named]
    lora_names = ([name for name,_ in backbone_named]
                  if args.backbone_training in ('full','qat','latent') else None)
    qat_named = backbone_named if args.backbone_training=='qat' else []
    backbone_lr = args.backbone_lr if args.backbone_training in ('full','latent') else args.lora_lr

    def policy_state():
        return backbone_trainable_state(backbone_named)
    if args.checkpoint_elide_rng:
        if not hasattr(backbone, 'elide_checkpoint_rng'):
            raise ValueError('checkpoint RNG elision is qualified only for the native LFM2 port')
        backbone.elide_checkpoint_rng()
    if args.crisp_weight and not lora:
        raise ValueError('crisp SFT requires a trainable backbone policy')
    # The writer's own modules (feedback, content projection) learn from the readers of what they write.
    # Native final normalization is a frozen reference. At full depth the
    # causal feedback is already exact; learn payload/stop without corrupting it.
    if args.read_adapter:
        heads.add_read_adapter()
    head_named = [(f'heads.{name}', p) for name, p in heads.named_parameters()
                   if not (not heads.read_markers and (name.startswith('feedback.final_norm.') or name.startswith('content.reference.') or
                           (heads.cutoff == backbone.num_layers and name.startswith('feedback.'))))] if args.heads_lr and (args.handover == "written" or args.view == "written") else []
    head_params = [p for _, p in head_named]
    if args.train_control_rows:
        head_params += [backbone.control_rows]
        head_named += [('backbone.control_rows', backbone.control_rows)]
        if not getattr(backbone, 'tied', True):
            head_params += [backbone.control_head_rows]
            head_named += [('backbone.control_head_rows', backbone.control_head_rows)]
    for p in head_params:
        p.requires_grad_(True)
    latent_lrs = {}
    if args.qat_latent_lr and args.backbone_training == 'qat':
        from .adapters import qat_latent_scales
        latent_lrs = {name: args.qat_latent_lr * scale for name, scale in qat_latent_scales(engine.backbone).items()
                      if name in set(lora_names or [])}
        print(json.dumps({'event': 'qat_latent_groups', 'latents': len(latent_lrs),
                          'lr_min': min(latent_lrs.values(), default=None),
                          'lr_max': max(latent_lrs.values(), default=None)}), flush=True)
    optimizer = trajectory_optimizer(args.optimizer, params, lora, head_params,
                                     vocab_size=backbone.embedding_weight.shape[0], lr=args.lr,
                                     lora_lr=backbone_lr, heads_lr=args.heads_lr,
                                     embedding_ids={id(backbone.control_rows), id(getattr(backbone, 'control_head_rows', backbone.control_rows))} | {id(p) for m in heads.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters()},
                                     lora_names=lora_names, latent_lrs=latent_lrs)
    from .optim_restore import optimizer_param_names, restore_optimizer_state, declared_added_names, record_restore_report
    optimizer_named = {**{f'soft.{name}': value for name, value in params.items()},
                       **({f'backbone.{name}': value for name, value in zip(lora_names, lora)} if lora_names is not None
                          else {f'lora_{i}': value for i, value in enumerate(lora)}),
                       **dict(head_named)}
    if resumed is not None:
        if set(params) != set(resumed['params']):
            raise ValueError('recurrence soft-parameter names changed')
        with torch.no_grad():
            for name, value in resumed['params'].items():
                params[name].copy_(value.to(params[name]))
            for name, value in {**resumed.get('lora', {}), **resumed.get('backbone_trainables', {})}.items():
                parameters=dict(backbone.hf.named_parameters())
                if name not in parameters or parameters[name].shape != value.shape:
                    raise ValueError('resumed backbone parameter mismatch: '+name)
                parameters[name].copy_(value.to(parameters[name]))
        with torch.no_grad():
            backbone.control_rows.copy_(resumed['control_rows'].to(backbone.control_rows))
            if not getattr(backbone, 'tied', True):
                backbone.control_head_rows.copy_(resumed['control_head_rows'].to(backbone.control_head_rows))
        missing, unexpected = heads.load_state_dict(resumed['heads'], strict=False)
        if unexpected or any(not k.startswith('read_adapter.') for k in missing):
            raise ValueError(f'resumed heads differ beyond a new read adapter: {missing} {unexpected}')
        # Named restore (optim_restore): unchanged groups load exactly as before; a changed trainable set is refused
        # unless declared. A read adapter declared with --curriculum-change read_adapter counts as added; further
        # additions are declared with --optimizer-added PREFIX; --optimizer-state fresh is the explicit reset
        # (continuations only, so a resume never silently re-resets).
        declared_prefixes = list(args.optimizer_added) + (['heads.read_adapter.'] if 'read_adapter' in args.curriculum_change else [])
        restore_report = restore_optimizer_state(
            optimizer, resumed['optimizer'], optimizer_named, declared_added_names(optimizer_named, declared_prefixes),
            saved_names=resumed.get('optimizer_param_names'),
            fresh=args.optimizer_state == 'fresh' and new_continuation)
        record_restore_report(restore_report, out, source=args.continue_from or 'resume')
        init = {k: v.to(params[k]) for k, v in resumed['init'].items()}
    if new_continuation and args.content_residual_initialization == 'fresh-zero':
        from .trajectory_state import initialize_content_residual
        reset = initialize_content_residual(heads, optimizer)
        (out / 'curriculum-initialization.json').write_text(json.dumps({
            'schema': 'natlang.content-residual-initialization/1', 'reset_parameters': reset,
            'optimizer_reset_scope': reset, 'other_optimizer_and_rng_preserved': True,
            'parent_sha256': identity['continuation']['checkpoint_sha256'],
            'initial_content_projection_zero': all(bool(torch.count_nonzero(p) == 0) for p in heads.content.proj.parameters()),
            'semantic_qualification_inherited': False}, indent=2) + '\n')
    if not heads.read_markers:
        heads.configure_frozen_reference()

    def count_embedding(items, dimensions):
        return sum(1 if kind == 'tok' else dimensions[value] + 2 for kind, value in items)

    def geometry_plan(record):
        dimensions = {block: params[name].shape[0] for name, block in leaf_ids.items()}
        features, memo = [], {}
        def visit(current, depth=0, visiting=()):
            own = target_writes(current) if depth else set()
            chosen = [name for name in sorted(reads(current) | set(handover_notes(current)))
                      if name in producers and name not in own and name not in visiting]
            if args.max_writes and len(chosen) > args.max_writes:
                chosen = sorted(write_choice.sample(chosen, args.max_writes))
            names = {}
            for name in chosen:
                names[name] = placeholder(name)
                key = name, depth + 1
                if share_producers and key in memo:
                    dimensions[names[name]] = memo[key]
                    continue
                producer = producers[name]
                child_names = visit(producer, depth + 1, visiting + (name,)) if depth + 1 < args.write_depth else {}
                cache_key = name, tuple(sorted(child_names))
                if cache_key not in geometry_cache:
                    messages = render(producer['messages'], lambda n: {'type': 'neuralese', 'id': leaf_ids[n]},
                                      handover_notes(producer), child_names, child_names)
                    rendered = render_messages(messages, producer.get('tools'), engine._template, engine.specials)
                    items = session._items(rendered.segments, rendered.blocks, rendered.escape_nonce)
                    context = count_embedding(items, dimensions) + len(engine._tokens(site_prefix(producer))) + 1
                    vectors = source_length(producer_source(name, producer)) or heads.max_length
                    geometry_cache[cache_key] = context, vectors
                context, vectors = geometry_cache[cache_key]
                features.append((context, vectors))
                dimensions[names[name]] = vectors
                memo[key] = vectors
            return names
        # Planning samples the identical edges without consuming training RNG.
        state = write_choice.getstate()
        try:
            names = visit(record)
            prompt, target = session._target_items(soft_messages(record, names), record.get('tools'), target_of(record, names))
            reader_context = count_embedding(prompt, dimensions) + count_embedding(target, dimensions)
            target_tokens = sum(kind == 'tok' for kind, _ in target)
        finally:
            write_choice.setstate(state)
        writer_bytes = sum(memory_estimator.predict('writer', context, vectors,
                           producer_geometry_bytes(context, vectors, memory_layout, native_gold=native_gold_tape,
                                                   vocab_size=backbone.embedding_weight.shape[0])) for context, vectors in features)
        reader_raw = geometry_bytes(reader_context, 0, **memory_layout, target_tokens=target_tokens,
                                    vocab_size=backbone.embedding_weight.shape[0])
        reader_bytes = memory_estimator.predict('reader', reader_context, target_tokens, reader_raw)
        from .memory_estimator import local_stage_kv_workspace
        max_kv_width = max([2 * backbone.layers[i].self_attn.k_proj.out_features
                            for i in range(backbone.num_layers) if backbone.is_attention(i)] or [0])
        workspace = max([local_stage_kv_workspace(c, n, memory_layout['stage_group_size'],
                         max_layer_kv_width=max_kv_width, dtype_bytes=memory_layout['dtype_bytes'])
                         for c, n in features] or [0])
        return {'writers': features, 'reader_context': reader_context, 'target_tokens': target_tokens,
                'reader_raw': reader_raw, 'tape_bytes': writer_bytes + reader_bytes,
                'branch_kv_workspace_bytes': workspace}

    reader_geometry = [None]
    offload_stats = {'offloaded_bytes': 0, 'live_offloaded_bytes': 0, 'peak_offloaded_bytes': 0}
    def retained_tape_bytes():
        # The predictor admits joint graphs without offload. Include CPU-stored
        # tape in observations instead of teaching it a falsely smaller cost.
        return cuda_allocated_bytes() + offload_stats.get('live_offloaded_bytes', 0)
    def observe_writer(value, retained_bytes):
        if args.staged_checkpoint_attention_only:
            # Staged replays can use different tapes. Never contaminate the
            # all-checkpointed joint admission profile with these measurements.
            return
        context, vectors = write_context_lengths[-1], value.shape[0]
        raw = producer_geometry_bytes(context, vectors, memory_layout, native_gold=native_gold_tape,
                                      vocab_size=backbone.embedding_weight.shape[0])
        memory_estimator.observe('writer', context, vectors, raw, retained_bytes)

    def purpose_contrast(record, base_leaves, names, leaves, messages, target):
        """View stage (data/view_records.py): the record's one view rewritten for a partner purpose over the same value
        (`view_stage.contrast_instructions`); hinge on the reader's target CE: the partner-purpose view must serve this
        reader at least --purpose-margin nats/token worse than the view written for its own purpose. The partner
        write's stop and boundary terms are not trained here (its length is not this record's choice)."""
        alternatives = (record.get("view_stage") or {}).get("contrast_instructions") or []
        parts = [p for m in record["messages"] for p in (m.get("content") if isinstance(m.get("content"), list) else [])
                 if p.get("type") == "view"]
        if not alternatives or len(parts) != 1:
            return None
        part = parts[0]
        stops, boundaries = len(stop_terms), len(boundary_terms)
        partner = dict(part, instructions=alternatives[0], name=part["name"] + "#contrast")
        partner.pop("faithful", None)
        payload = view_payload(record, partner, base_leaves)
        del stop_terms[stops:], boundary_terms[boundaries:]
        partner_names = dict(names)
        partner_names[part["name"]] = placeholder(partner["name"])
        partner_leaves = resolve_values({**leaves, partner_names[part["name"]]: payload})
        own = session.supervised_text_loss({"messages": messages, "tools": record.get("tools"), "target": target}, leaves)
        other = session.supervised_text_loss({"messages": soft_messages(record, partner_names), "tools": record.get("tools"),
                                              "target": target}, partner_leaves)
        contrast_values.append(float(other.detach() - own.detach()))
        return torch.relu(args.purpose_margin - (other - own))

    def loss_of(record, leaves, soft=True, training_objective=True):
        if not soft:
            crisp = crisp_messages(record["messages"], texts, handover_notes(record))
            target = render([record["target"]], lambda name: {"type": "text", "text": texts[name]}, handover_notes(record))[0]
            return session._term({"kind": "crossEntropy", "messages": crisp, "tools": record.get("tools"), "target": target}, leaves)
        # Notes and views are written afresh by the current writer; their payloads are leaves of this loss.
        stop_terms.clear()
        boundary_terms.clear()
        base_leaves = leaves
        names, payloads = written_values(record, leaves)
        leaves = resolve_values({**leaves, **payloads})
        messages, target = soft_messages(record, names), target_of(record, names)
        distill = args.distill if training_objective and payloads and not target_write(record) else 0
        reader_before = retained_tape_bytes() if args.device.startswith('cuda') else 0
        loss = session.supervised_text_loss(
            {"messages": messages, "tools": record.get("tools"), "target": target}, leaves,
            teacher_messages=crisp_messages(record["messages"], texts, handover_notes(record)) if distill else None,
            distill_weight=distill, context_weight=args.context_weight if training_objective else 0.,
            feedback_weight=args.feedback_weight,
            covered_replies=coverage.get(record['id']) if args.context_coverage == 'records' else None,
            projection_anchor_weight=anchor_now[0] if training_objective else 0.,
            projection_anchor_backbone_scale=args.projection_anchor_backbone_scale)
        if training_objective and anchor_now[0]:
            projection_anchor_values.append(float(session.last_projection_anchor_loss))
        if training_objective and args.purpose_contrast and args.view == "written":
            contrast = purpose_contrast(record, base_leaves, names, leaves, messages, target)
            if contrast is not None:
                loss = loss + args.purpose_contrast * contrast
        if reader_geometry[0] and torch.is_grad_enabled() and args.device.startswith('cuda'):
            plan = reader_geometry[0]
            memory_estimator.observe('reader', plan['reader_context'], plan['target_tokens'], plan['reader_raw'],
                                     max(0, retained_tape_bytes() - reader_before))
        if not training_objective:
            # Evaluation reports pure target CE, not CE plus distillation,
            # stop-boundary penalties or policy-gradient terms.
            boundary_terms.clear()
            stop_terms.clear()
            return loss
        if active_staging[0] is not None:
            loss = loss + active_staging[0].penalty_loss(1.)
        if boundary_terms:
            loss = loss + sum(boundary_terms) / len(boundary_terms)
            boundary_terms.clear()
        if stop_terms:
            # Stop policy (phase E's objective on real readers): reward = -(reader loss + λ·length), against a running
            # baseline; the stop head learns how long a note or view must be for what its readers need.
            reward = -(float(loss.detach()) + args.stop_pg * sum(n for _, n in stop_terms))
            advantage = reward - (baseline["value"] if baseline["value"] is not None else reward)
            baseline["value"] = reward if baseline["value"] is None else 0.9 * baseline["value"] + 0.1 * reward
            loss = loss - advantage * sum(logp for logp, _ in stop_terms)
            stop_terms.clear()
        return loss

    def evaluate(label, leaves, soft=True, records=None):
        values = []
        with torch.no_grad():
            for record in _progress(held if records is None else records, label):
                try:
                    values.append(float(loss_of(record, leaves, soft, training_objective=False)))
                except RequestError:
                    pass
        return {"label": label, "cross_entropy": sum(values) / max(1, len(values)), "n": len(values)}

    def evaluate_written(label, leaves, records, accounting):
        """Pure reader CE with own and proven factual-disjoint typed donor blocks.

        Donors are fixed before execution. A failed writer never changes donor
        selection, drops an eligible row silently, or cycles one block into slots.
        """
        stop_terms.clear()
        boundary_terms.clear()
        sites, write_errors, reader_errors, missing_donors = {}, [], [], []
        probe_memo = ProducerMemo(share_producers)
        with torch.no_grad():
            for record in _progress(records, label):
                try:
                    names, payloads = written_values(record, leaves, memo=probe_memo, reader_only=True)
                    if not payloads:
                        raise RequestError('paired_reader_missing_payload', 'selected paired reader produced no read payloads')
                    sites[record['id']] = (record, names, payloads)
                except RequestError as error:
                    write_errors.append({'id': record['id'], 'error': str(error)})
        result = {'label': label, 'policy': probe_policy, 'n': 0, 'expected_n': len(records),
                  'selection_sha256': probe_selection_hash,
                  'coverage': accounting, 'case_scores': [],
                  'producer_cache_hits': probe_memo.hits, 'producer_cache_entries': len(probe_memo.values),
                  'write_errors': write_errors, 'reader_errors': reader_errors, 'missing_donors': missing_donors,
                  'qualification_scope': probe_selection['scope']}
        own, shuffled, groups = [], [], {}
        with torch.no_grad():
            for record, names, payloads in sites.values():
                rid = record['id']
                plan = accounting['selected_mappings'][rid]
                donor_id = plan['donor_id']
                if donor_id not in sites:
                    missing_donors.append({'id': rid, 'donor_id': donor_id, 'reason': 'fixed donor write failed'})
                    continue
                donor_record, donor_names, donor_payloads = sites[donor_id]
                mapping = plan['donor_payload_to_recipient_payload']
                if set(mapping) != set(donor_names) or set(mapping.values()) != set(names):
                    raise RuntimeError('paired reader slot proof disagrees with actual resolved reads')
                swapped = {names[recipient]: donor_payloads[donor_names[donor]]
                           for donor, recipient in mapping.items()}
                if set(swapped) != set(payloads):
                    raise RuntimeError('paired reader payload bijection changed during execution')
                def reader(values):
                    return float(session._term({'kind': 'crossEntropy', 'messages': soft_messages(record, names),
                                                'tools': record.get('tools'), 'target': target_of(record, names)},
                                               {**leaves, **values}))
                try:
                    a, b = reader(payloads), reader(swapped)
                except RequestError as error:
                    reader_errors.append({'id': rid, 'error': str(error)})
                    continue
                own.append(a)
                shuffled.append(b)
                factual = sorted(source_groups(record))
                key = tuple(factual)
                groups.setdefault(key, []).append((a, b))
                result['case_scores'].append({'id': rid, 'source_groups': factual, 'donor_id': donor_id,
                    'donor_source_groups': sorted(source_groups(donor_record)), 'payload_mapping': mapping,
                    'written': a, 'shuffled': b, 'margin': b - a,
                    'slot_lengths': [{'recipient': recipient, 'donor': donor,
                                     'own': int(payloads[names[recipient]].shape[0]),
                                     'donor_length': int(donor_payloads[donor_names[donor]].shape[0])}
                                    for donor, recipient in sorted(mapping.items())]})
        stop_terms.clear()
        boundary_terms.clear()
        if own:
            group_means = [{'source_groups': list(key), 'rows': len(values),
                            'written': sum(a for a, _ in values) / len(values),
                            'shuffled': sum(b for _, b in values) / len(values)}
                           for key, values in sorted(groups.items())]
            result.update({'written': sum(own) / len(own), 'shuffled': sum(shuffled) / len(shuffled),
                           'n': len(own), 'written_better': sum(a < b for a, b in zip(own, shuffled)) / len(own),
                           'factual_group_set_means': group_means,
                           'factual_group_set_written_mean': sum(g['written'] for g in group_means)/len(group_means),
                           'factual_group_set_shuffled_mean': sum(g['shuffled'] for g in group_means)/len(group_means)})
        return result

    leaves = {leaf_ids[name]: p for name, p in params.items()}
    report = dict(resumed['initial_report']) if resumed is not None else {"crisp": evaluate("crisp", {}, soft=False), "soft-init": evaluate("soft-init", leaves)}
    if resumed is None and family and args.member_eval:
        report["family-init"] = evaluate_members(backbone, member_eval_windows())
    if resumed is None and cohort_strata_held:
        # Other cohorts' held records: reported strata, never qualification evidence.
        report["cohort_strata-init"] = {
            cohort: {"crisp": evaluate(f"crisp-{cohort}", {}, soft=False, records=chosen),
                     "soft": evaluate(f"soft-init-{cohort}", leaves, records=chosen)}
            for cohort, chosen in cohort_strata_held.items()}
    if resumed is None and (args.handover == "written" or args.view == "written"):
        report["written-init"] = evaluate_written("written-init", leaves, paired_held, held_probe_accounting)
        report["written-init-train"] = evaluate_written("written-init-train", leaves, paired_train, train_probe_accounting)
    if resumed is not None and resumed.get('probe_selection_sha256') != probe_selection_hash:
        # Evaluate the new scope before restoring RNG state below. Historical
        # initial scores remain explicitly historical, not re-labelled as new.
        baseline_eval = {'completed_updates': resumed['step'], 'event': 'probe_scope_baseline',
                         'selection_sha256': probe_selection_hash,
                         'soft': evaluate('resume-scope-soft', leaves)}
        if args.handover == 'written' or args.view == 'written':
            baseline_eval['written'] = evaluate_written('resume-scope-written', leaves, paired_held, held_probe_accounting)
        report['resume_scope_baseline'] = baseline_eval
        report['historical_initial_scope'] = 'Earlier initial reports retain their original selection scope.'
        with (out / 'eval.jsonl').open('a') as baseline_log:
            baseline_log.write(json.dumps(baseline_eval) + '\n')
    print(json.dumps(report), flush=True)
    log = open(out / "train.jsonl", "a" if resumed is not None else "w")
    started, cursor, errors, used = time.time(), 0, 0, set()
    start_step = 0
    if resumed is not None:
        start_step, cursor, errors, used = resumed['step'], resumed['cursor'], resumed['errors'], set(resumed['used'])
    # The anchor decays from where this stage began: a resume keeps its origin, a continuation starts a new one.
    anchor_origin = (resumed.get('anchor_origin', start_step) if resumed is not None and not new_continuation
                     else start_step)
    def scheduled_anchor(step):
        if not args.projection_anchor_decay_steps:
            return args.projection_anchor_weight
        return args.projection_anchor_weight * max(0.0, 1.0 - (step - anchor_origin) / args.projection_anchor_decay_steps)
    if resumed is not None:
        baseline.update(resumed['baseline'])
        random.setstate(resumed['python_rng'])
        write_choice.setstate(resumed['write_rng'])
        stop_generator.set_state(resumed['stop_rng'])
        torch.set_rng_state(resumed['torch_rng'])
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(resumed['cuda_rng'])
    graph_routes = dict(resumed.get('graph_routes', {})) if resumed and memory_estimator.joint_routes_compatible else {}
    trainables = list(params.values()) + lora + head_params
    from .loop import (Cadence, StopSignal, TrainingLoop, accumulate_gradients,
                       commit_optimizer_step)
    stop = StopSignal().install()
    from .checkpoint_policy import publish_best
    # Declared step points only (owner 2026-10-10; plans/STORAGE_POLICY.md): the initial report is the gate's reference
    # (fresh lineages only), evaluations every --eval-every updates, full-state writes every --checkpoint-every
    # updates (evaluation points), at the end and on stops; the final report is the stage end's evaluation.
    eval_cadence, checkpoint_cadence = Cadence(args.eval_every), Cadence(args.checkpoint_every)
    from .trajectory_state import compatible_best_evaluation
    selection_signature = {'files': identity['files'], 'port_profile': heads.profile,
                           'content_transport': heads.content.transport, 'sketch_gradient': args.sketch_gradient,
                           'writer_length_policy': args.writer_length_policy, 'writer_supervision': args.writer_supervision, 'stop_supervision': args.stop_supervision,
                           'tokens_per_vector': args.tokens_per_vector, 'max_write_vectors': heads.max_length,
                           'write_depth': args.write_depth, 'max_writes': args.max_writes,
                           'write_curriculum': args.write_curriculum, 'probe_policy': probe_policy,
                           'probe_selection_sha256': probe_selection_hash}
    inherited_best = resumed.get('best_evaluation') if resumed else None
    best_evaluation = compatible_best_evaluation(inherited_best, selection_signature)
    best_history = list(resumed.get('best_evaluation_history', [])) if resumed else []
    if inherited_best is not None and best_evaluation is None:
        if inherited_best not in best_history:
            best_history.append(inherited_best)
        print(json.dumps({'status': 'candidate_selection_regime_changed',
                          'inherited_candidate_preserved_in_history': True,
                          'selection_signature': selection_signature}), flush=True)
    host_gc_seconds = 0.0
    host_gc_calls = 0
    host_gc_generation_stats = {0: {'calls': 0, 'seconds': 0.0, 'unreachable': 0},
                                2: {'calls': 0, 'seconds': 0.0, 'unreachable': 0}}
    phase_wall_seconds = {}
    pending_cuda_phase_events = []

    def start_phase(name, step):
        started = time.perf_counter()
        event = None
        if args.device.startswith('cuda'):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return {'name': name, 'step': step, 'started': started, 'cuda_start': event}

    def stop_phase(phase):
        phase_wall_seconds[phase['name']] = phase_wall_seconds.get(phase['name'], 0.0) + (
            time.perf_counter() - phase['started'])
        if phase['cuda_start'] is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            pending_cuda_phase_events.append((phase['step'], phase['name'], phase['cuda_start'], end))

    def flush_cuda_phase_events():
        if not pending_cuda_phase_events:
            return
        by_step = {}
        for step_index, name, start, end in pending_cuda_phase_events:
            phases = by_step.setdefault(step_index, {})
            phases[name] = phases.get(name, 0.0) + start.elapsed_time(end)
        with (out / 'cuda-phase-timings.jsonl').open('a') as timing_log:
            for step_index in sorted(by_step):
                timing_log.write(json.dumps({
                    'schema': 'natlang.neuralese-recurrence-cuda-phase-timings/1',
                    'step': step_index,
                    'milliseconds': by_step[step_index],
                    'measurement': 'CUDA events on the current stream; resolved after checkpoint state copies complete; no per-phase synchronization',
                    'overlap': 'record_model_and_autograd contains staged_producer_replay when staging is selected'
                }) + '\n')
        pending_cuda_phase_events.clear()

    def collect_graph_cycles(generation=2):
        # Host wall time only: do not synchronize CUDA. Generation 0 is for
        # short-lived per-writer graphs; generation 2 remains at phase edges.
        nonlocal host_gc_seconds, host_gc_calls
        started_gc = time.perf_counter()
        collected = 0
        try:
            collected = gc.collect(generation)
            return collected
        finally:
            elapsed = time.perf_counter() - started_gc
            host_gc_seconds += elapsed
            host_gc_calls += 1
            stats = host_gc_generation_stats.setdefault(
                generation, {'calls': 0, 'seconds': 0.0, 'unreachable': 0})
            stats['calls'] += 1
            stats['seconds'] += elapsed
            stats['unreachable'] += collected

    def collect_local_graph_cycles():
        return collect_graph_cycles(0)

    def save_training_state(step, destination=None):
        started_save = time.perf_counter()
        destination = destination or checkpoint_path
        state_snapshot = {
            'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'identity': identity, 'graph_routes': graph_routes, 'memory_estimator': memory_estimator.state_dict(),
            'execution_policy': {'checkpoint_layers': args.checkpoint_layers,
                                 'checkpoint_preserve_rng': getattr(backbone, 'checkpoint_preserve_rng', True),
                                 'producer_batch_size': args.producer_batch_size,
                                 'joint_producer_batching': args.joint_producer_batching,
                                 'token_cache_mib': args.token_cache_mib,
                                 'producer_batch_memory_gb': args.producer_batch_memory_gb,
                                 'checkpoint_attention_only': args.checkpoint_attention_only,
                                 'staged_checkpoint_attention_only': args.staged_checkpoint_attention_only,
                                 'activation_offload_gb': args.activation_offload_gb,
                                 'geometry_version': geometry_version},
            'artifact_refs': input_roles,
            'step': step, 'cursor': cursor, 'errors': errors, 'used': sorted(used), 'best_evaluation': best_evaluation, 'best_evaluation_history': best_history,
            'params': {k: v.detach().cpu() for k, v in params.items()}, 'texts': texts,
            'control_rows': backbone.control_rows.detach().cpu(),
            **({'control_head_rows': backbone.control_head_rows.detach().cpu()} if not getattr(backbone, 'tied', True) else {}),
            'port_config': {'cutoff': heads.cutoff, 'max_length': heads.max_length, **heads.port_config()},
            'heads': heads.state_dict(), 'lora': lora_state(backbone), 'optimizer': optimizer.state_dict(),
            'optimizer_param_names': optimizer_param_names(optimizer, optimizer_named),
            'backbone_training': args.backbone_training,
            **({'maple_qat': True} if qat_named else {}),
            **({'backbone_trainables': policy_state()} if args.backbone_training in ('full','qat','latent') else {}), 'anchor_origin': anchor_origin,
            'init': {k: v.detach().cpu() for k, v in init.items()}, 'initial_report': report,
            'probe_selection_sha256': probe_selection_hash,
            'baseline': baseline, 'python_rng': random.getstate(), 'write_rng': write_choice.getstate(),
            'stop_rng': stop_generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else []}
        snapshot_ready = time.perf_counter()
        atomic_checkpoint(destination, state_snapshot)
        checkpoint_committed = time.perf_counter()
        # Snapshot construction includes device-to-host copies; atomic commit
        # includes serialization and durable I/O. Both are blocking wall times.
        phase = {'event': 'checkpoint_saved', 'completed_updates': step,
                 'path': str(destination),
                 'blocking_seconds': checkpoint_committed - started_save,
                 'snapshot_and_device_copy_wall_seconds': snapshot_ready - started_save,
                 'serialization_and_durable_io_wall_seconds': checkpoint_committed - snapshot_ready,
                 'measurement': 'monotonic wall clock; snapshot construction includes host/device copies'}
        with (out / 'host-phases.jsonl').open('a') as phase_log:
            phase_log.write(json.dumps(phase) + '\n')
        # The checkpoint snapshot's device-to-host copies have completed the
        # queued current-stream events. Resolve events here, never at each phase.
        flush_cuda_phase_events()
    save_training_state(start_step)
    # The model, fixed corpus, optimizer and imported libraries live until this
    # dedicated trainer process exits. Collect completed setup/probe cycles first,
    # then exclude this static population from every staged graph collection.
    # Dynamic training graphs created below remain tracked/collectible. Do not
    # unfreeze an existing library baseline or freeze after an update's graph.
    setup_collected = gc.collect()
    frozen_before = gc.get_freeze_count()
    gc.freeze()
    print(json.dumps({'event': 'static_gc_baseline_frozen', 'setup_collected': setup_collected,
                      'previous_frozen_objects': frozen_before, 'frozen_objects': gc.get_freeze_count(),
                      'scope': 'process-lifetime setup only; future graph collection unchanged'}), flush=True)
    with torch.enable_grad():
        # A stop is observed after a step has run and been checkpointed, never before the first one.
        loop = TrainingLoop(start_step, args.steps, stop, stop_before_step=False)
        for step in loop:
            anchor_now[0] = scheduled_anchor(step)
            # One precision point per update (train/quantization.py, sampled by weight; BF16 teachers unaffected).
            if quant is not None:
                step_precision = quant.sampled(step)
                quant.set_active(*step_precision[:2])
            step_started = time.perf_counter()
            phase_wall_seconds = {}
            step_gc_seconds, step_gc_calls = host_gc_seconds, host_gc_calls
            step_gc_generation_stats = {g: dict(v) for g, v in host_gc_generation_stats.items()}
            step_lengths_start = len(lengths)
            step_batches_start = len(writer_batches)
            step_contexts_start = len(write_context_lengths)
            selective_writer_replays[0] = 0
            step_cursor, step_errors, step_used = cursor, errors, set(used)
            step_rng = iteration_rng_state(write_choice, stop_generator, baseline, cuda=args.device.startswith('cuda'))
            optimizer.zero_grad(set_to_none=True)
            losses = []
            crisp_losses = []
            projection_anchor_values.clear()
            contrast_values.clear()
            view_lengths.clear()
            released_graph_bytes = 0
            offload_stats = {'offloaded_bytes': 0, 'live_offloaded_bytes': 0, 'peak_offloaded_bytes': 0}
            staged_nodes, replay_error = 0, 0.0
            discarded_writer_batches, discarded_writer_rows = 0, 0
            step_peak_bytes = 0
            step_record_ids = []
            for _ in range(args.batch):
                if args.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                record = train[cursor % len(train)] if sampler is None else sampler.record(cursor)
                step_record_ids.append(record['id'])
                cursor += 1
                geometry_phase_started = time.perf_counter()
                mode = args.backward_policy
                plan = geometry_plan(record) if args.backward_policy == 'auto' else None
                reader_geometry[0] = plan
                if mode == 'auto':
                    baseline_bytes = cuda_allocated_bytes() if args.device.startswith('cuda') else 0
                    raw_prediction = baseline_bytes + plan['tape_bytes'] + plan['branch_kv_workspace_bytes']
                    predicted = memory_estimator.adjust_joint(plan, raw_prediction)
                    mode = 'staged' if predicted > graph_budget or graph_routes.get(record['id'], 0) >= graph_budget else 'joint'
                    with (out / 'memory-routing.jsonl').open('a') as routing_log:
                        routing_log.write(json.dumps({'step': step, 'record_id': record['id'], 'mode': mode,
                            'estimated_gib': predicted / 2**30, 'graph_budget_gib': graph_budget / 2**30,
                            'write_sites': len(plan['writers']), 'writer_geometry': plan['writers'],
                            'sketch_gradient': args.sketch_gradient, 'local_stage_batch_size': args.local_stage_batch_size,
                            'branch_kv_workspace_gib': plan['branch_kv_workspace_bytes'] / 2**30}) + '\n')
                phase_wall_seconds['geometry_and_router_wall'] = (
                    phase_wall_seconds.get('geometry_and_router_wall', 0.0) +
                    time.perf_counter() - geometry_phase_started)
                rng_before = (random.getstate(), write_choice.getstate(), stop_generator.get_state(),
                              torch.get_rng_state(), torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                              dict(baseline), len(lengths), len(write_context_lengths), len(writer_batches),
                              len(projection_anchor_values))
                def attempt_joint():
                    # autograd.grad avoids partial parameter .grad mutations on
                    # an aborted attempt, including accumulated earlier chains.
                    try:
                        with graph_memory_budget(graph_budget):
                            objective = loss_of(record, leaves) / args.batch
                            gradients = torch.autograd.grad(objective, trainables, allow_unused=True)
                        peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
                        return float(objective.detach()), gradients, None, peak
                    except (GraphBudgetExceeded, torch.OutOfMemoryError) as failure:
                        peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
                        return None, None, type(failure).__name__, peak
                record_phase = start_phase('record_model_and_autograd', step)
                try:
                    if args.backward_policy == 'auto' and mode == 'joint':
                        value, gradients, failure, joint_peak = attempt_joint()
                        memory_estimator.observe_joint(plan, raw_prediction, joint_peak, failed=bool(failure))
                        if failure:
                            with (out / 'memory-routing.jsonl').open('a') as routing_log:
                                routing_log.write(json.dumps({'event': 'joint_failure', 'step': step,
                                    'record_id': record['id'], 'failure': failure,
                                    'observed_failed_peak_gib': joint_peak / 2**30,
                                    'estimated_gib': predicted / 2**30,
                                    'local_stage_batch_size': args.local_stage_batch_size}) + '\n')
                            graph_routes[record['id']] = int(graph_budget) + 1
                            random.setstate(rng_before[0])
                            write_choice.setstate(rng_before[1])
                            stop_generator.set_state(rng_before[2])
                            torch.set_rng_state(rng_before[3])
                            if rng_before[4]:
                                torch.cuda.set_rng_state_all(rng_before[4])
                            baseline.clear()
                            baseline.update(rng_before[5])
                            del lengths[rng_before[6]:]
                            del write_context_lengths[rng_before[7]:]
                            discarded_writer_batches += len(writer_batches) - rng_before[8]
                            discarded_writer_rows += sum(writer_batches[rng_before[8]:])
                            del writer_batches[rng_before[8]:]
                            del projection_anchor_values[rng_before[9]:]
                            boundary_terms.clear()
                            stop_terms.clear()
                            collect_graph_cycles()
                            if args.device.startswith('cuda'):
                                torch.cuda.empty_cache()
                            mode = 'staged'
                            print(json.dumps({'status': 'stage_for_budget', 'record_id': record['id'],
                                              'joint_failure': failure, 'graph_budget_gib': graph_budget / 2**30}), flush=True)
                        else:
                            accumulate_gradients(trainables, gradients)
                            del gradients
                            losses.append(value * args.batch)
                    if mode == 'staged' or args.backward_policy == 'joint':
                        active_staging[0] = StagedWrites(
                            observe=observe_writer,
                            measure=retained_tape_bytes if args.device.startswith('cuda') else None,
                            collect=collect_graph_cycles,
                            collect_local=collect_local_graph_cycles) if mode == 'staged' else None
                        from .memory import offload_attention_tensors
                        persistent = list(backbone.parameters()) + list(backbone.buffers()) + list(heads.parameters()) + list(heads.buffers()) + list(params.values())
                        with offload_attention_tensors(int(args.activation_offload_gb * 2**30), activations=True,
                                                      persistent_tensors=persistent) as offload_stats:
                            loss = loss_of(record, leaves) / args.batch
                            loss.backward()
                            losses.append(float(loss.detach()) * args.batch)
                            del loss
                            collect_graph_cycles()
                            if active_staging[0] is not None:
                                replay_phase = start_phase('staged_producer_replay', step)
                                try:
                                    active_staging[0].backward(penalty_weight=1., scale=1 / args.batch)
                                finally:
                                    stop_phase(replay_phase)
                                staged_nodes += len(active_staging[0].nodes)
                                replay_error = max(replay_error, active_staging[0].replay_max_abs_error)
                                active_staging[0].clear()
                                active_staging[0] = None
                    # The policy learns from the exact ordinary runtime too. Its graph
                    # is built after the recurrent graph has been freed; all gradients
                    # accumulate before one optimizer step, never doubling live tapes.
                    if args.crisp_weight:
                        objective = args.crisp_weight * loss_of(record, {}, soft=False) / args.batch
                        gradients = torch.autograd.grad(objective, trainables, allow_unused=True)
                        accumulate_gradients(trainables, gradients)
                        crisp_losses.append(float(objective.detach()) * args.batch)
                        del objective, gradients
                    collect_graph_cycles()
                    case_peak = torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
                    step_peak_bytes = max(step_peak_bytes, case_peak)
                    if args.backward_policy == 'auto':
                        with (out / 'memory-routing.jsonl').open('a') as routing_log:
                            routing_log.write(json.dumps({'event': 'result', 'step': step, 'record_id': record['id'],
                                'mode': mode, 'observed_peak_gib': case_peak / 2**30,
                                'estimated_gib': predicted / 2**30}) + '\n')
                except RequestError as error:
                    stop_phase(record_phase)
                    if active_staging[0] is not None:
                        active_staging[0].clear()
                        active_staging[0] = None
                    errors += 1
                    # A skipped record contributes no gradient; say which and why (it was only counted).
                    print(json.dumps({'status': 'record_request_error', 'step': step, 'record_id': record['id'],
                                      'error': str(error)[:500]}), flush=True)
                    continue
                except (RuntimeError, AssertionError, ValueError) as error:
                    stop_phase(record_phase)
                    failure = {'status': 'training_out_of_memory' if isinstance(error, torch.OutOfMemoryError) else 'training_pre_update_failure',
                               'error_type': type(error).__name__, 'error': str(error),
                               'step': step, 'record_id': record['id'],
                               'curriculum': args.write_curriculum, 'batch': args.batch, 'backward_mode': mode,
                               'offload': dict(offload_stats)}
                    # No optimizer update occurs inside this accumulation loop.
                    # Keep all preceding updates and rewind this incomplete step,
                    # including every sampler, so resumption repeats its inputs.
                    optimizer.zero_grad(set_to_none=True)
                    if active_staging[0] is not None:
                        active_staging[0].clear()
                        active_staging[0] = None
                    cursor, errors, used = step_cursor, step_errors, step_used
                    restore_iteration_rng(step_rng, write_choice, stop_generator, baseline)
                    save_training_state(step)
                    failure['emergency_checkpoint'] = str(checkpoint_path)
                    failure['checkpoint_step'] = step
                    (out / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
                    print(json.dumps(failure), flush=True)
                    raise
                stop_phase(record_phase)
                used.update(part["name"] for m in record["messages"] if isinstance(m.get("content"), list)
                            for part in m["content"] if part["type"] == "soft")
            reader_geometry[0] = None
            family_record = None
            if args.member_weight and step_record_ids:
                # The family term (MAPLE_NESTED §4a): one member per update, in rotation, on this update's last record.
                member = family[step % len(family)]
                parts = member_backward(backbone, member, *member_window(record), weight=args.member_weight,
                                        full_weight=args.member_full_weight)
                family_record = {'member': member.key, 'ce': parts.ce / max(parts.tokens, 1),
                                 'kl': parts.kl / max(parts.tokens, 1), 'tokens': parts.tokens}
            # The writer's gradient from its readers: zero would mean written values do not train the writer.
            if preserve is not None:
                # Behaviour preservation at this update's precision (train/quantization.py).
                preserve_loss, preserved = preserve.loss(engine.backbone.hf, step)
                preserve_loss.backward()
                del preserve_loss
            optimizer_phase = start_phase('gradient_clip_and_optimizer', step)
            writer_grad = float(gradient_norm(head_params)) if head_params else None
            clip_finite_gradients(trainables, 1.0)
            if quant is not None:
                quant.set_active('bf16', 0.0)  # evaluations and checkpoints after the update see BF16
            commit_optimizer_step(optimizer)
            stop_phase(optimizer_phase)
            entry = {"step": step, "iteration_index": step, "completed_updates": step + 1, "reader_record_ids": step_record_ids, "loss": sum(losses) / max(1, len(losses)), "seconds": round(time.time() - started),
                     "errors": errors, "backward_mode": mode, "staged_nodes": staged_nodes,
                     "host_gc_seconds": round(host_gc_seconds - step_gc_seconds, 6),
                     "host_gc_calls": host_gc_calls - step_gc_calls,
                     "host_gc_by_generation": {
                         str(g): {
                             'calls': host_gc_generation_stats[g]['calls'] - step_gc_generation_stats.get(g, {}).get('calls', 0),
                             'seconds': round(host_gc_generation_stats[g]['seconds'] - step_gc_generation_stats.get(g, {}).get('seconds', 0.0), 6),
                             'unreachable': host_gc_generation_stats[g]['unreachable'] - step_gc_generation_stats.get(g, {}).get('unreachable', 0),
                         } for g in sorted(host_gc_generation_stats)
                     },
                     "replay_max_abs_error": replay_error, "selective_writer_replays": selective_writer_replays[0], "crisp_sft_loss": sum(crisp_losses) / max(1, len(crisp_losses)), "step_seconds": round(time.perf_counter() - step_started, 3), **({"writer_grad_norm": writer_grad} if head_params else {}),
                     **({"write_lengths": lengths[step_lengths_start:][-8:]} if len(lengths) > step_lengths_start else {}),
                     "writes_this_update": len(lengths) - step_lengths_start,
                     "writer_batch_calls_this_update": len(writer_batches) - step_batches_start,
                     "writer_batch_rows_this_update": sum(writer_batches[step_batches_start:]),
                     "max_writer_batch_rows_this_update": max(writer_batches[step_batches_start:], default=1),
                     "discarded_writer_batches_this_update": discarded_writer_batches,
                     "discarded_writer_rows_this_update": discarded_writer_rows,
                     "phase_wall_seconds": dict(phase_wall_seconds),
                     **({"projection_anchor_loss": sum(projection_anchor_values) / len(projection_anchor_values)}
                        if projection_anchor_values else {}),
                     **({"purpose_contrast_gap": sum(contrast_values) / len(contrast_values)} if contrast_values else {}),
                     **({"view_writes_this_update": len(view_lengths), "max_view_length_this_update": max(view_lengths)}
                        if view_lengths else {}),
                     "max_write_length_this_update": max(lengths[step_lengths_start:], default=0),
                     "write_capacity": heads.max_length, **({"family": family_record} if family_record else {})}
            if quant is not None:
                entry['precision'] = {'point': step_precision[0], 'mix': step_precision[1],
                                      **(preserved if preserve is not None else {})}
            if args.device.startswith("cuda"):
                entry["peak_gb"] = round(max(step_peak_bytes, torch.cuda.max_memory_allocated()) / 2**30, 2)
                entry['released_graph_gib'] = round(released_graph_bytes / 2**30, 3)
                entry['largest_write_context_tokens'] = max(write_context_lengths[step_contexts_start:], default=0)
                entry['activation_offloaded_gib'] = round(offload_stats['offloaded_bytes'] / 2**30, 3)
                entry['activation_offload_peak_gib'] = round(offload_stats['peak_offloaded_bytes'] / 2**30, 3)
            if args.token_cache_mib:
                entry['token_cache'] = engine._token_cache.stats()
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 10 == 0 or step == args.steps - 1:
                print(json.dumps(entry), flush=True)
            write_due = checkpoint_cadence.due(step + 1, force=stop.requested or step + 1 == args.steps)
            candidate = None
            # Evaluations at the declared points; the last step's is the final report below.
            if not stop.requested and step + 1 < args.steps and eval_cadence.due(step + 1):
                from .trajectory_state import evaluation_state
                with evaluation_state(write_choice, stop_generator, baseline):
                    evaluation = {'step': step + 1, 'soft': evaluate('periodic-soft', leaves)}
                    if quant is not None:
                        evaluation['quantization'] = {**quant.describe(step + 1), 'precision': step_precision[0],
                                                      'columns': {}}
                        if behaviour is not None:
                            evaluation['quantization']['bf16_behaviour'] = behaviour.report(engine.backbone.hf)
                        for point in quant.gated_points():
                            with quant.context(point, 1.0):
                                evaluation['quantization']['columns'][point] = evaluate(f'periodic-soft-{point}', leaves)
                                if behaviour is not None:
                                    measured = behaviour.report(engine.backbone.hf)
                                    evaluation['quantization']['columns'][point + ':behaviour'] = {
                                        **measured, 'passed': behaviour.passed(measured)}
                    if codes is not None:
                        evaluation['qat_codes'] = codes.update()
                    if family and args.member_eval:
                        evaluation['family'] = evaluate_members(backbone, member_eval_windows())
                    if args.crisp_weight:
                        evaluation['crisp'] = evaluate('periodic-crisp', {}, soft=False)
                    if args.handover == 'written' or args.view == 'written':
                        evaluation['written'] = evaluate_written('periodic-written', leaves, paired_held, held_probe_accounting)
                with (out / 'eval.jsonl').open('a') as evaluation_log:
                    evaluation_log.write(json.dumps(evaluation) + '\n')
                print(json.dumps({'evaluation': evaluation}), flush=True)
                written = evaluation.get('written', {})
                score = written.get('written')
                from .trajectory_state import paired_probe_complete
                if (score is not None and paired_probe_complete(written)
                        and written.get('shuffled', score) > score
                        and (best_evaluation is None or score < best_evaluation['written'])):
                    candidate = {'step': step + 1, **written, 'semantic_channel_qualified': False,
                                 'selection_signature': selection_signature, 'checkpoint': str(out / 'best-checkpoint.pt'),
                                 'selection_scope': 'candidate by complete paired reader CE; separate semantic/stopping eval required'}
            if write_due:
                # Full state only at declared points, on a stop and at the end (owner 2026-10-10). The best is one of
                # these writes: when this point's evaluation is the best so far, the same file is linked as
                # best-checkpoint.pt (the end's candidate comes from the final report below).
                if candidate is not None and not stop.requested:
                    best_evaluation = candidate
                save_training_state(step + 1)
                if candidate is not None and not stop.requested:
                    publish_best(checkpoint_path, out / 'best-checkpoint.pt')
                    (out / 'best-evaluation.json').write_text(json.dumps(best_evaluation, indent=2) + '\n')
    stop.restore()
    log.close()
    if stop.requested:
        print(json.dumps({'status': 'checkpointed_on_signal', 'checkpoint': str(checkpoint_path)}), flush=True)
        return 0
    report["soft-trained"] = evaluate("soft-trained", leaves)
    if quant is not None:
        report["quantization"] = {**quant.describe(args.steps), "columns": {}}
        for point in quant.gated_points():
            with quant.context(point, 1.0):
                report["quantization"]["columns"][point] = evaluate(f"soft-trained-{point}", leaves)
                if behaviour is not None:
                    measured = behaviour.report(engine.backbone.hf)
                    report["quantization"]["columns"][point + ":behaviour"] = {**measured,
                                                                               "passed": behaviour.passed(measured)}
    if cohort_strata_held:
        report["cohort_strata-trained"] = {cohort: {"soft": evaluate(f"soft-trained-{cohort}", leaves, records=chosen)}
                                           for cohort, chosen in cohort_strata_held.items()}
    if family and args.member_eval:
        report["family-trained"] = evaluate_members(backbone, member_eval_windows())
    if args.handover == "written" or args.view == "written":
        report["written-trained"] = evaluate_written("written-trained", leaves, paired_held, held_probe_accounting)
        report["written-trained-train"] = evaluate_written("written-trained-train", leaves, paired_train, train_probe_accounting)
        # The stage end's evaluation selects among the full-state writes too: the end write holds these weights.
        written = report["written-trained"]
        score = written.get('written')
        from .trajectory_state import paired_probe_complete
        if (score is not None and paired_probe_complete(written) and written.get('shuffled', score) > score
                and (best_evaluation is None or score < best_evaluation['written']) and checkpoint_path.exists()):
            best_evaluation = {'step': args.steps, **written, 'semantic_channel_qualified': False,
                               'selection_signature': selection_signature,
                               'checkpoint': str(out / 'best-checkpoint.pt'),
                               'selection_scope': 'candidate by complete paired reader CE; separate semantic/stopping eval required'}
            publish_best(checkpoint_path, out / 'best-checkpoint.pt')
            (out / 'best-evaluation.json').write_text(json.dumps(best_evaluation, indent=2) + '\n')
    if lengths:
        report["writes"] = {"count": len(lengths), "mean_length": sum(lengths) / len(lengths), "max_length": max(lengths)}
    if head_params or lora:
        from .adapters import adapter_layers
        source_metadata = torch.load(args.heads, map_location='cpu', weights_only=False, mmap=True) if args.heads else {}
        ranks = {int(p.shape[0]) for name, p in backbone.hf.named_parameters() if '.lora_A.' in name}
        if len(ranks) > 1:
            raise ValueError('cannot export mixed-rank adapters in one port checkpoint')
        atomic_checkpoint(out / 'heads.pt', {'heads': heads.state_dict(),
            'port_config': {**source_metadata.get('port_config', {}), 'cutoff': heads.cutoff,
                            'max_length': heads.max_length, **heads.port_config()},
            'control_rows': backbone.control_rows.detach().cpu(),
            **({'control_head_rows': backbone.control_head_rows.detach().cpu()} if not getattr(backbone, 'tied', True) else {}), 'lora': lora_state(backbone),
            'lora_layers': adapter_layers(backbone), 'lora_rank': next(iter(ranks), 0),
            'backbone_training': args.backbone_training,
            **({'maple_qat': True} if qat_named else {}),
            **({'backbone_trainables': policy_state()} if args.backbone_training in ('full','qat','latent') else {}), 'anchor_origin': anchor_origin,
            'backbone': source_metadata.get('backbone') or {'base': args.base},
            'training_identity': identity})
    # Every soft parameter's movement: also those only producers' contexts hold, which move by their readers' losses.
    moved = {name: float((params[name].detach() - init[name]).norm() / init[name].norm().clamp_min(1e-9)) for name in params}
    report["relative_change"] = moved
    torch.save({"params": {k: v.detach().cpu() for k, v in params.items()}, "texts": texts,
                'port_profile': heads.profile}, out / "soft-params.pt")
    if lora:
        torch.save({"lora": lora_state(engine.backbone), "rank": args.rank}, out / "adapter.pt")
    if bank:
        current = {name.removeprefix("prompt:"): params[name] for name in used if name.removeprefix("prompt:") in bank.texts}
        save_bank(out / "system-prompts.nz", bank, current, {"kind": "system-prompt-bank", "init": args.bank,
                  "trained_by": "natlang_neuralese.train.trajectories", "records": args.records, "steps": args.steps})
    (out / "summary.json").write_text(json.dumps({"options": vars(args), **report}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "relative_change"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
