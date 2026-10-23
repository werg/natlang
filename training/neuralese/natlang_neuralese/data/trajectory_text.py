"""Static text and trajectory-IR conversion helpers.

These helpers authenticate and render recorded trajectory structure for ordinary
text preparation and native graph conversion. They do not create model-side
Neuralese vectors or run training.
"""
from __future__ import annotations

import hashlib
import json
import re
from ..train.write_sites import extract_write_sites, extract_write_sites_from_calls
from ..train.inline_instructions import validate_inline_instruction_code
from ..view import reject_digest_part, view_note

INSTRUCTIONS = re.compile(r"Instructions:\n([\s\S]*?)\n\n(?:In eval|Eval also|$)")


def render(messages: list[dict], soft_part, notes: dict[str, str], *, neuralese_bodies=None) -> list[dict]:
    """Expand recorded message parts to ordinary text using authenticated source bodies.

    Opaque native states must first be converted through a native graph/source
    conversion. This renderer intentionally has no opaque-block injection path.
    """
    bodies = neuralese_bodies or {}
    out = []
    for original in messages:
        message = dict(original)
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                kind = part.get("type")
                if kind == "soft":
                    rendered = soft_part(part["name"])
                    if isinstance(rendered, str):
                        parts.append(rendered)
                    elif isinstance(rendered, dict) and rendered.get("type") == "text" and isinstance(rendered.get("text"), str):
                        parts.append(rendered["text"])
                    else:
                        raise ValueError("ordinary trajectory rendering requires text for each soft part")
                elif kind == "view":
                    parts.append(part.get("preview", ""))
                elif kind == "digest":
                    reject_digest_part(part)
                elif kind == "read":
                    name = part.get("name")
                    parts.append(part.get("source", notes.get(name, "")))
                elif kind == "neuralese":
                    body = bodies.get(part.get("id"))
                    if not isinstance(body, str):
                        raise ValueError("opaque native state requires authenticated graph/source conversion")
                    parts.append(body)
                elif kind == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                else:
                    raise ValueError("trajectory text renderer received an unsupported content part")
            message["content"] = "".join(parts)
        calls = message.get("tool_calls")
        if calls:
            rendered_calls = []
            for call in calls:
                function = call.get("function") or {}
                arguments = function.get("arguments")
                if isinstance(arguments, list):
                    text = []
                    for part in arguments:
                        if not isinstance(part, dict):
                            raise ValueError("tool-call argument part is not an object")
                        if part.get("type") == "text" and isinstance(part.get("text"), str):
                            text.append(part["text"])
                        elif part.get("type") == "read" and isinstance(part.get("source"), str):
                            text.append(json.dumps(part["source"], ensure_ascii=False)[1:-1])
                        else:
                            raise ValueError("native or unresolved tool-call argument needs graph conversion")
                    arguments = "".join(text)
                if not isinstance(arguments, str):
                    raise ValueError("tool-call arguments are not text")
                if "neuralese_code" in call:
                    if not isinstance(arguments, str):
                        raise ValueError("inline instruction sidecar needs text arguments")
                    from ..train.inline_instructions import render_inline_instruction_arguments
                    arguments = render_inline_instruction_arguments(arguments, call["neuralese_code"])
                    if arguments is None:
                        raise ValueError("invalid inline instruction code sidecar")
                    call = {key: value for key, value in call.items() if key != "neuralese_code"}
                if '"$write"' in arguments:
                    def crisp_value(value):
                        if isinstance(value, dict) and "$write" in value:
                            site = value["$write"]
                            source = site["source"]
                            json_body = site.get("source_encoding") == "json" or site.get("type") == "Neuralese<unknown>"
                            return json.loads(source) if json_body else source
                        if isinstance(value, dict):
                            return {key: crisp_value(item) for key, item in value.items()}
                        if isinstance(value, list):
                            return [crisp_value(item) for item in value]
                        return value
                    arguments = json.dumps(crisp_value(json.loads(arguments)), ensure_ascii=False)
                rendered_calls.append({**call, "function": {**function, "arguments": arguments}})
            message["tool_calls"] = rendered_calls
        out.append(message)
    return out


def crisp_messages(messages: list[dict], texts: dict[str, str], notes: dict[str, str], *, neuralese_bodies=None) -> list[dict]:
    return render(messages, lambda name: texts[name], notes, neuralese_bodies=neuralese_bodies)


def native_writer_prefix(record, apply_template, *, value_type=None):
    """Render the exact recorded typed-return prefix for static text conversion."""
    from ..serve.chat import write_reply, write_value_text
    call, before, argument, name = write_site(record)
    if call == 'assistant-text-return':
        return apply_template(record['messages'], True)
    value_type = value_type or write_value_type(record)
    path = write_value_path(record, name)
    arguments = write_site_arguments(record, name) if len(path) > 1 else before
    path_arg = path if len(path) > 1 else None
    prefix = write_reply(apply_template, call, arguments, argument, value_type, argument_path=path_arg)[0]
    code_prefix = inline_write_prefix(record, name)
    if code_prefix is not None:
        prefix += write_value_text(apply_template, call, before, argument, code_prefix, value_type)
    return prefix


def producer_text_target(record, texts, names):
    """Render a recorded producer's gold reply as ordinary text for text-only preparation."""
    own = target_writes(record)
    ancestors = {name: body for name, body in names.items() if name not in own}
    return render([record['target']], lambda name: texts[name], handover_notes(record),
                  neuralese_bodies=ancestors)[0]

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

def write_sites(record: dict) -> list[tuple[str, dict, str, str]]:
    """Legacy tuple view over the shared validated target write-site extractor."""
    parsed = extract_write_sites(record)
    names=[site.name for site in parsed]
    if len(names)!=len(set(names)):
        raise ValueError('duplicate writer name in target')
    return [(site.tool,dict(site.before_arguments),site.argument,site.name) for site in parsed]

def write_value_path(record: dict, name: str | None = None) -> tuple[str | int, ...]:
    """Exact argument-tree path for one uniquely named writer in the target."""
    selected=name or record.get('_active_write_name')
    found=[site for site in extract_write_sites(record)
           if (selected is not None and site.name==selected) or
              (selected is None and site.source_kind=='json-write-marker')]
    if len(found)!=1:
        if not found and selected is not None and inline_write_prefix(record,selected) is not None:
            return ()
        raise ValueError('writer name is absent or has multiple value paths')
    return found[0].legacy_path

def write_site_arguments(record: dict, name: str | None = None) -> dict:
    """Complete native call arguments for a writer, used when its value is nested."""
    selected=name or record.get('_active_write_name')
    sites=extract_write_sites(record)
    if selected is None:
        if sites:return dict(sites[0].arguments)
        raise ValueError('writer arguments are absent')
    matches=[site for site in sites if site.name==selected]
    if len(matches)!=1:
        if len(matches)>1:raise ValueError('writer name is ambiguous in target')
        raise ValueError('writer arguments are absent')
    if matches[0].source_kind == 'assistant-text-sidecar':
        return {}
    return dict(matches[0].arguments)

def write_site(record: dict) -> tuple[str, dict, str, str] | None:
    selected=record.get('_active_write_name')
    return next((site for site in write_sites(record) if selected is None or site[3]==selected),None)

def inline_write_prefix(record: dict, name: str) -> str | None:
    return next((site.inline_prefix for site in extract_write_sites(record)
                 if site.name==name and site.source_kind=='inline-code-sidecar'),None)

def authenticated_capture_context_augmentation(record: dict, writer_name: str) -> tuple[dict | None, str | None]:
    """Expose authenticated full captures only as an explicit, separately hashable training augmentation."""
    from ..train.inline_instructions import validate_inline_instruction_code
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
    parsed_sites=extract_write_sites(record)
    names=[site.name for site in parsed_sites]
    if len(names)!=len(set(names)):raise ValueError('duplicate writer name in target')
    selected=record.get('_active_write_name')
    parsed=next((item for item in parsed_sites if selected is None or item.name==selected),None)
    if parsed is None:return 'string'
    if parsed.source_kind=='inline-code-sidecar':return 'string'
    kind=parsed.value_type
    if kind=='Neuralese<string>':return 'string'
    if kind=='Neuralese<unknown>' or parsed.descriptor.get('source_encoding')=='json':return 'unknown'
    # A named Neuralese type whose crisp value is text (the compiler's Neuralese<HandoverNote> compact_history note,
    # or any writer whose argument schema is a string) is a string write.
    if kind=='Neuralese<HandoverNote>' or (isinstance(parsed.schema,dict) and parsed.schema.get('type')=='string'):
        return 'string'
    raise ValueError(f'nested writer has unsupported declared value type {kind!r}')

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
        calls = [call for call in (message.get("tool_calls") or [])
                 if "neuralese_code" in call or
                 (isinstance((call.get("function") or {}).get("arguments"), str) and
                  '"$write"' in call["function"]["arguments"])]
        for site in extract_write_sites_from_calls(calls):
            remember(site.name, site.source)
    for site in extract_write_sites(record):
        if site.source_kind == 'assistant-text-sidecar':
            remember(site.name, site.source)
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

def merge_context_reviews(reviews: list) -> dict:
    """One recurrence-context review of several separately authenticated record groups (files or trajectories)."""
    merged = {"schema": "natlang.recurrence-context-review/1", "groups": len(reviews)}
    for key in ("source_records", "prepared_records", "excluded_context_records"):
        merged[key] = sum(review[key] for review in reviews)
    for key in ("prepared_by_split", "excluded_by_split"):
        merged[key] = {split: sum(review[key].get(split, 0) for review in reviews) for split in ("train", "test")}
    for key in ("decisions", "excluded"):
        merged[key] = [entry for review in reviews for entry in review[key]]
    return merged
