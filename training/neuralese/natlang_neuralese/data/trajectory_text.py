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
    """Authenticate model-neutral occurrence ledgers before graph compilation.

    New conversion rows contain plain text and exact path-bound source
    occurrences. This reader binds each consumer occurrence to one producer in
    the same trajectory; the tokenizer-specific graph compiler later replaces
    every exact consumer span with positional imports.
    """
    import copy
    import hashlib
    import re
    from ..common.jsonio import canonical_json_str

    def digest(value):
        return hashlib.sha256(canonical_json_str(value).encode("utf-8")).hexdigest()

    def occurrence_path(row, path):
        current = row
        for part in path:
            if part.endswith("<json>"):
                field=part[:-6]
                raw=(current.get(field) if isinstance(current,dict) else current)
                if not isinstance(raw,str):
                    return None
                try: current = json.loads(raw)
                except json.JSONDecodeError: return None
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                if index >= len(current): return None
                current = current[index]
            elif isinstance(current, dict):
                if part not in current: return None
                current = current[part]
            else:
                return None
        return current

    rows = list(records)
    producers = {}
    metadata_by_id = {}
    source_hashes = {}
    for row in rows:
        rid = row.get("id")
        meta = row.get("native_source_conversion")
        if not isinstance(meta, dict) or meta.get("schema") != "natlang.native-source-conversion/1":
            raise ValueError(f"source row {rid} lacks the current native occurrence conversion")
        if (meta.get("row_id") != rid or not isinstance(meta.get("source_record_sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", meta["source_record_sha256"]) or
                not isinstance(meta.get("source_messages_sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", meta["source_messages_sha256"]) or
                meta.get("source_messages_sha256") != digest(meta.get("source_messages")) or
                meta.get("source_target_sha256") != digest(meta.get("source_target")) or
                meta.get("native_target_sha256") != digest(row.get("target"))):
            raise ValueError(f"native source conversion hash binding is malformed for {rid}")
        original = dict(row)
        original.pop("native_source_conversion", None)
        original["messages"] = meta["source_messages"]
        if meta.get("source_target_present") is True:
            original["target"] = meta["source_target"]
        else:
            original.pop("target", None)
        if meta.get("source_neuralese_conversion_present") is True:
            original["neuralese_conversion"] = meta.get("source_neuralese_conversion")
        else:
            original.pop("neuralese_conversion", None)
        if digest(original) != meta["source_record_sha256"]:
            raise ValueError(f"native source conversion does not reconstruct its exact source record for {rid}")
        source_ref = row.get("source_ref") or {}
        if (meta.get("source_ref") != source_ref or
                meta.get("trajectory_id") != source_ref.get("trajectory_id") or
                meta.get("invocation_id") != source_ref.get("invocation_id") or
                meta.get("source_trajectory_row_sha256") != source_ref.get("source_row_sha256") or
                meta.get("source_provenance") != row.get("provenance", {}) or
                meta.get("training_admission") != row.get("training_admission") or
                meta.get("split") != row.get("split") or
                meta.get("source_groups") != row.get("source_groups")):
            raise ValueError(f"native source conversion actor/partition/admission binding changed for {rid}")
        occurrences = meta.get("occurrences")
        if not isinstance(occurrences, list):
            raise ValueError(f"native source conversion occurrences are missing for {rid}")
        unresolved = meta.get("unresolved_references")
        if not isinstance(unresolved, list):
            raise ValueError(f"native source conversion unresolved-reference ledger is malformed for {rid}")
        if unresolved:
            raise ValueError(f"native source conversion has unresolved exact source references for {rid}")
        seen = set()
        for occurrence in occurrences:
            if (not isinstance(occurrence, dict) or occurrence.get("schema") != "natlang.native-source-occurrence/1" or
                    occurrence.get("role") not in {"producer", "consumer"} or
                    not isinstance(occurrence.get("id"), str) or not re.fullmatch(r"nso_[0-9a-f]{64}", occurrence["id"]) or
                    occurrence["id"] in seen or not isinstance(occurrence.get("reference"), str) or
                    not occurrence["reference"] or not isinstance(occurrence.get("path"), list) or
                    any(not isinstance(part, str) for part in occurrence["path"]) or
                    not isinstance(occurrence.get("body"), str) or
                    occurrence.get("body_sha256") != hashlib.sha256(occurrence["body"].encode("utf-8")).hexdigest() or
                    occurrence.get("trajectory_id") != meta.get("trajectory_id") or
                    occurrence.get("invocation_id") != meta.get("invocation_id")):
                raise ValueError(f"native source occurrence is malformed for {rid}")
            seen.add(occurrence["id"])
            if occurrence["role"] == "consumer":
                value = occurrence_path(row, occurrence["path"])
                if not isinstance(value, dict) or value.get("type") != "text" or value.get("text") != occurrence["body"]:
                    raise ValueError(f"native consumer source path/body mismatch for {rid}:{occurrence['id']}")
            else:
                if not occurrence["path"] or occurrence["path"][0] not in {"target", "source_target", "source_receipts"}:
                    raise ValueError(f"native producer occurrence is outside its exact target/source receipt for {rid}")
                root = (row if occurrence["path"][0] == "target" else
                        {"source_target": meta.get("source_target")} if occurrence["path"][0] == "source_target" else
                        {"source_receipts": meta.get("source_receipts")})
                value = occurrence_path(root, occurrence["path"])
                exact = value == occurrence["body"]
                if (not exact and occurrence["role"] == "producer"
                        and occurrence["path"][-1:] == ["$write"]
                        and isinstance(value, dict)):
                    # Typed tool-result records bind the producer occurrence
                    # at the exact `$write` descriptor. Authenticate its
                    # declared type, event name, and source bytes instead of
                    # requiring the descriptor object itself to equal text.
                    exact = (value.get("type") == "Neuralese<string>"
                             and value.get("name") == occurrence.get("reference")
                             and value.get("source") == occurrence["body"])
                if not exact and isinstance(value, (dict, list)):
                    try:
                        exact = json.loads(occurrence["body"]) == value
                    except json.JSONDecodeError:
                        exact = False
                if not exact:
                    raise ValueError(f"native producer source path/body mismatch for {rid}:{occurrence['id']}")
                prior = producers.get(occurrence["reference"])
                if prior is not None:
                    raise ValueError("duplicate native source producer reference: " + occurrence["reference"])
                producers[occurrence["reference"]] = (row, occurrence)
        metadata_by_id[rid] = (meta, occurrences)
        source_hashes[rid] = meta["source_record_sha256"]

    prepared, decisions, excluded = [], [], []
    for row in rows:
        rid = row.get("id")
        meta, occurrences = metadata_by_id[rid]
        split = row.get("split")
        groups = sorted(set(group for group in row.get("source_groups", []) if isinstance(group, str) and group))
        refs = []
        for consumer in (item for item in occurrences if item["role"] == "consumer"):
            writer_pair = producers.get(consumer["reference"])
            body = consumer["body"]
            ref = {"kind": "native-positional-state", "name": consumer["reference"],
                   "body_sha256": consumer["body_sha256"], "reader_occurrence_id": consumer["id"],
                   "reader_path": consumer["path"], "gradient_edge": False}
            if writer_pair:
                writer_row, writer = writer_pair
                writer_groups = set(writer_row.get("source_groups") or [])
                same_source = ((writer_row.get("source_ref") or {}).get("source_row_sha256") and
                               (writer_row.get("source_ref") or {}).get("source_row_sha256") ==
                               (row.get("source_ref") or {}).get("source_row_sha256"))
                approved = ((writer_row.get("training_admission") or {}).get("approved") is True and
                            (row.get("training_admission") or {}).get("approved") is True)
                exact = (writer["body"] == body and writer["body_sha256"] == consumer["body_sha256"] and
                         writer_row.get("split") == split and
                         (writer_groups == set(row.get("source_groups") or []) or same_source) and approved)
                if exact:
                    ref.update(writer_record_id=writer_row.get("id"), writer_occurrence_id=writer["id"],
                               writer_path=writer["path"], source_kind=writer["source_kind"], gradient_edge=True)
                else:
                    ref["reason"] = "writer body, split, source-group or admission binding differs"
            else:
                ref["reason"] = "no exact selected producer occurrence in this trajectory"
            refs.append(ref)
        prepared.append(copy.deepcopy(row))
        decisions.append({"id": rid, "source_record_sha256": source_hashes[rid], "split": split,
                          "source_groups": groups, "status": "eligible", "references": refs,
                          "unresolved_references": meta.get("unresolved_references", [])})
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
