"""Pure-data safeguards for selecting held readers and aligned write donors.

This module deliberately does not import the trainer or evaluate model outputs.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


def _program_aliases(record: Mapping[str, Any]) -> set[str]:
    aliases = set()
    for value in (record.get("program_id"), (record.get("source_ref") or {}).get("program_ir_id")):
        if isinstance(value, str) and value:
            aliases.add(value)
    ids = record.get("source_program_ids") or []
    if isinstance(ids, str):
        ids = [ids]
    aliases.update(x for x in ids if isinstance(x, str) and x)
    return aliases


def source_groups(record: Mapping[str, Any]) -> frozenset[str]:
    """Return factual groups, excluding program identity aliases.

    Remaining groups are authoritative provenance and are not rewritten or
    inferred from task text.
    """
    groups = record.get("source_groups") or []
    if isinstance(groups, str):
        groups = [groups]
    aliases = _program_aliases(record)
    return frozenset(x for x in groups if isinstance(x, str) and x and x not in aliases)


def _reads(record: Mapping[str, Any]) -> set[str]:
    own = {w["write"].get("name") for w in _target_writes(record)}
    return {part["name"] for message in record.get("messages", [])
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "read" and isinstance(part.get("name"), str) and part["name"] not in own}


def select_held(records: Iterable[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    """Select deterministically, round-robin by factual group.

    Actual read sites are queued before records without reads. Records with
    multiple factual groups participate in the lexically first group, avoiding
    duplicate selection while keeping the policy stable across runs.
    """
    if limit <= 0:
        return []
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups = sorted(source_groups(record))
        group = groups[0] if groups else "<unattributed>"
        buckets[group].append(record)
    for group in buckets:
        buckets[group].sort(key=lambda r: (not bool(_reads(r)), str(r.get("id", ""))))
    selected = []
    groups = sorted(buckets)
    while groups and len(selected) < limit:
        remaining = []
        for group in groups:
            if buckets[group] and len(selected) < limit:
                selected.append(buckets[group].pop(0))
            if buckets[group]:
                remaining.append(group)
        groups = remaining
    return selected


def _target_writes(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    def visit(value: Any, path: tuple[str | int, ...]):
        if isinstance(value, dict) and isinstance(value.get("$write"), dict):
            out.append({"argument": path[0] if path else None, "value_path": path,
                        "write": value["$write"]})
        elif isinstance(value, dict):
            for key, item in value.items():visit(item,path+(key,))
        elif isinstance(value, list):
            for index, item in enumerate(value):visit(item,path+(index,))
    for call in ((record.get("target") or {}).get("tool_calls") or []):
        try:
            args = json.loads(call["function"]["arguments"])
        except (KeyError, TypeError, ValueError):
            continue
        before={}
        for arg,value in args.items():
            start=len(out);visit(value,(arg,))
            for site in out[start:]:site["tool"]=call["function"].get("name")
            before[arg]=value
    return out


def _strip_descriptions(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_descriptions(v) for k, v in value.items() if k != "description"}
    if isinstance(value, list):
        return [_strip_descriptions(v) for v in value]
    return value


def _arg_schema(record: Mapping[str, Any], tool_name: str | None, argument: str,
                value_path: Iterable[str | int] | None = None) -> Any:
    for tool in record.get("tools", []) or []:
        fn = tool.get("function", {})
        if fn.get("name") == tool_name:
            schema=(fn.get("parameters") or {}).get("properties", {}).get(argument)
            for part in tuple(value_path or ())[1:]:
                if not isinstance(schema,dict):return None
                schema=(schema.get("properties",{}).get(part) if isinstance(part,str) else
                        schema.get("items") if type(part) is int else None)
            return _strip_descriptions(schema)
    return None


def _function_body_ids(record: Mapping[str, Any], piece_kinds: Mapping[str, Any]) -> tuple[str, ...]:
    found = set()
    for message in record.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "soft":
                continue
            name = part.get("name")
            kind = piece_kinds.get(name)
            if isinstance(kind, Mapping):
                kind = kind.get("kind")
            if kind == "function-body":
                found.add(name)
    return tuple(sorted(found))


def _site_record(site: Any, producers: Mapping[str, Mapping[str, Any]]) -> tuple[Mapping[str, Any], list[str]]:
    if isinstance(site, Mapping) and isinstance(site.get("record"), Mapping):
        rec = site["record"]
        names = site.get("payload_names") or sorted(_reads(rec))
    else:
        rec = site
        names = sorted(_reads(rec))
    return rec, sorted(set(name for name in names if isinstance(name, str)))


def aligned_donor(recipient: Any, candidates: Iterable[Any], producers: Mapping[str, Mapping[str, Any]],
                  piece_kinds: Mapping[str, Any]) -> dict[str, Any]:
    """Find a donor with a proven one-to-one output-slot alignment.

    `recipient` and candidate sites are records or `{record, payload_names}`
    mappings. `producers` maps payload ID to its producer record. Success maps
    each donor payload ID to the corresponding recipient payload ID.
    """
    rr, rnames = _site_record(recipient, producers)
    rgroups = source_groups(rr)
    if not rgroups:
        return {"donor": None, "mapping": {}, "reason": "recipient has no factual source group"}
    if not rnames:
        return {"donor": None, "mapping": {}, "reason": "recipient has no resolved read payloads"}
    def producer_slots(names: list[str], reader: Mapping[str, Any]) -> tuple[dict[str, tuple], dict[str, dict]] | str:
        signatures, writes_by_name = {}, {}
        for name in names:
            producer = producers.get(name)
            write_rows=_target_writes(producer or {})
            write_names=[w["write"].get("name") for w in write_rows]
            if len(write_names)!=len(set(write_names)):
                return "producer has ambiguous writer names"
            writes = {w["write"].get("name"): w for w in write_rows}
            if producer is None or name not in writes:
                return "read payload has no target-write producer proof"
            body = _function_body_ids(producer, piece_kinds)
            if not body:
                return "producer has no function-body soft IDs"
            w = writes[name]
            schema = _arg_schema(producer, w["tool"], w["argument"],w.get("value_path"))
            if not isinstance(schema, dict):
                return "producer has no argument schema proof"
            native = w["write"].get("source")
            if not isinstance(native, str):
                return "producer has no native source text"
            pgroups = source_groups(producer)
            if not pgroups or not pgroups.issubset(source_groups(reader)):
                return "producer factual groups are absent or not contained in reader groups"
            signatures[name] = (w["tool"], w["argument"], tuple(w.get("value_path",())), schema, body)
            writes_by_name[name] = w
        return signatures, writes_by_name

    rproof = producer_slots(rnames, rr)
    if isinstance(rproof, str):
        return {"donor": None, "mapping": {}, "reason": "recipient " + rproof}
    rsig, rwrites = rproof
    failures = []
    for candidate in candidates:
        dr, dnames = _site_record(candidate, producers)
        dgroups = source_groups(dr)
        if not dgroups or rgroups & dgroups:
            failures.append("candidate source groups missing or overlap")
            continue
        if len(dnames) != len(rnames) or not dnames:
            failures.append("candidate slot count differs")
            continue
        if set(dnames) & set(rnames):
            failures.append("candidate shares payload IDs")
            continue
        dproof = producer_slots(dnames, dr)
        if isinstance(dproof, str):
            failures.append("candidate " + dproof)
            continue
        dsig, dwrites = dproof
        mapping = {}
        unused = set(rnames)
        valid = True
        for dname in dnames:
            dw = dwrites[dname]["write"]
            matches = []
            for rname in sorted(unused):
                rw = rwrites[rname]["write"]
                if dsig[dname] == rsig[rname] and dw.get("source") != rw.get("source"):
                    matches.append(rname)
            if len(matches) != 1:
                valid = False
                break
            mapping[dname] = matches[0]
            unused.remove(matches[0])
        if valid and not unused and len(mapping) == len(rnames):
            return {"donor": dr.get("id"), "mapping": mapping, "reason": "aligned"}
        failures.append("no unique schema/body/source-text bijection")
    reason = (failures[0] if len(set(failures)) == 1 else
              "no aligned donor: multiple proof failures" if failures else "no candidates")
    return {"donor": None, "mapping": {}, "reason": reason, "candidate_rejections": len(failures),
            "candidate_rejection_reason_counts": dict(sorted(Counter(failures).items()))}


def select_paired_held(records: Iterable[Mapping[str, Any]], limit: int,
                       producers: Mapping[str, Mapping[str, Any]],
                       piece_kinds: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Pack mutually proven reader/donor pairs round-robin by factual group.

    Only rows with actual reads and a reverse-valid donor enter paired scope.
    Returned mappings are keyed by selected recipient ID. The row limit is
    strict; an odd final slot remains unused rather than breaking a pair.
    """
    rows = list(records)
    readers = [r for r in rows if _reads(r)]
    by_id = {str(r.get("id")): r for r in readers}
    eligible_pairs: dict[str, dict[str, Any]] = {}
    rejected = Counter()
    for rec in sorted(readers, key=lambda r: str(r.get("id", ""))):
        rid = str(rec.get("id"))
        if rid in eligible_pairs:
            continue
        for candidate in readers:
            cid = str(candidate.get("id"))
            if rid == cid or cid in eligible_pairs:
                continue
            forward = aligned_donor(rec, [candidate], producers, piece_kinds)
            if forward.get("donor") != cid:
                rejected[forward.get("reason", "no donor")] += 1
                continue
            reverse = aligned_donor(candidate, [rec], producers, piece_kinds)
            if reverse.get("donor") != rid or {v: k for k, v in forward["mapping"].items()} != reverse["mapping"]:
                rejected["reverse proof or inverse mapping failed"] += 1
                continue
            eligible_pairs[rid] = {"recipient": rec, "donor": candidate,
                                   "mapping": forward["mapping"], "reverse_mapping": reverse["mapping"]}
            eligible_pairs[cid] = {"recipient": candidate, "donor": rec,
                                   "mapping": reverse["mapping"], "reverse_mapping": forward["mapping"]}
            break
    pair_keys = sorted({tuple(sorted((rid, str(pair["donor"].get("id")))))
                        for rid, pair in eligible_pairs.items()})
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for pair in pair_keys:
        groups = sorted(source_groups(by_id[pair[0]]) | source_groups(by_id[pair[1]]))
        buckets[groups[0] if groups else "<unattributed>"].append(pair)
    for group in buckets:
        buckets[group].sort()
    selected_ids: list[str] = []
    selected_pairs = []
    groups = sorted(buckets)
    while groups and len(selected_ids) + 2 <= max(0, limit):
        remaining = []
        for group in groups:
            if buckets[group] and len(selected_ids) + 2 <= max(0, limit):
                pair = buckets[group].pop(0)
                selected_pairs.append(pair)
                selected_ids.extend(pair)
            if buckets[group]:
                remaining.append(group)
        groups = remaining
    mappings = {}
    for recipient_id, donor_id in selected_pairs:
        item = eligible_pairs[recipient_id]
        mappings[recipient_id] = {"donor_id": donor_id, "donor_payload_to_recipient_payload": item["mapping"]}
        mappings[donor_id] = {"donor_id": recipient_id, "donor_payload_to_recipient_payload": item["reverse_mapping"]}
    reason_by_row, proof_by_row = {}, {}
    for rec in readers:
        rid = str(rec.get("id"))
        if rid not in eligible_pairs:
            sample = [c for c in readers if c is not rec]
            proof = aligned_donor(rec, sample, producers, piece_kinds)
            reason_by_row[rid] = ("compatible donor excluded by disjoint pair packing" if proof.get('donor')
                                  else proof.get("reason", "no donor"))
            proof_by_row[rid] = proof
    family_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    for rid in selected_ids:
        rec = by_id[rid]
        task = rec.get("task")
        program_family = (((task.get("program_ir") or {}).get("family")) if isinstance(task, Mapping) else None)
        family_counts[str(rec.get("task_family") or program_family or rec.get("family") or
                          ((rec.get("curriculum") or {}).get("family")) or "unknown")] += 1
        for group in source_groups(rec):
            group_counts[group] += 1
    accounting = {
        "input_rows": len(rows), "reader_rows": len(readers), "eligible_reader_rows": len(eligible_pairs),
        "eligible_pairs": len(pair_keys), "selected_rows": len(selected_ids), "selected_pairs": len(selected_pairs),
        "limit": max(0, limit), "eligible_pair_ids": [list(p) for p in pair_keys],
        "selected_ids": selected_ids, "selected_mappings": mappings,
        "pairing_strategy": "deterministic greedy disjoint reciprocal pairs; not maximum matching",
        "reader_factual_group_counts": dict(sorted(Counter(g for r in readers for g in source_groups(r)).items())),
        "eligible_factual_group_counts": dict(sorted(Counter(g for rid in eligible_pairs for g in source_groups(by_id[rid])).items())),
        "selected_factual_group_counts": dict(sorted(group_counts.items())),
        "selected_task_family_counts": dict(sorted(family_counts.items())),
        "ineligible_reader_count": len(readers) - len(eligible_pairs),
        "ineligible_reason_counts": dict(sorted(Counter(reason_by_row.values()).items())),
        "ineligible_reason_by_id": reason_by_row,
        "ineligible_proof_by_id": proof_by_row,
        "pair_search_rejection_counts": dict(sorted(rejected.items())),
    }
    return [by_id[rid] for rid in selected_ids], accounting
