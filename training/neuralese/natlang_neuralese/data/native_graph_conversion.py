"""Exact static conversion of admitted token-aligned rows into native graph rows.

This module is intentionally separate from source policy: callers must supply
role spans backed by source receipts/witnesses. It never infers producer or
consumer roles from marker-like strings in a flat transcript.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping

from ..common.hashing import canonical_json_sha256_hex
from .native_graph import SCHEMA


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_row(row: Mapping[str, Any], *, source_tokenizer_fingerprint: str,
                  tokenizer, expanded_tokenizer_fingerprint: str):
    if row.get("tokenizer_sha256") != source_tokenizer_fingerprint:
        raise ValueError("text row tokenizer differs from its pinned source tokenizer")
    if tokenizer is None or not hasattr(tokenizer, "decode"):
        raise ValueError("exact source-token decoding is required for native graph conversion")
    if type(row.get("supervised_suffix_start")) is not int:
        raise ValueError("text row needs an exact supervised suffix boundary")
    ids = row.get("token_ids")
    if (not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids)
            or not 0 <= row["supervised_suffix_start"] < len(ids)):
        raise ValueError("text row token IDs or supervised boundary are invalid")
    text = row.get("text")
    if not isinstance(text, str) or tokenizer.decode(ids, skip_special_tokens=False) != text:
        raise ValueError("text row IDs do not decode to its exact rendered source text")
    if not isinstance(expanded_tokenizer_fingerprint, str) or len(expanded_tokenizer_fingerprint) != 64:
        raise ValueError("the expanded role-token tokenizer fingerprint must be pinned")
    split = row.get("split")
    groups = row.get("source_groups")
    if split not in {"train", "test"} or not isinstance(groups, list) or not groups:
        raise ValueError("text row needs an explicit split and source groups")
    if any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("text row has invalid source groups")
    identifier = row.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("text row needs its source identity")
    return ids, text, split, sorted(set(groups)), identifier


def text_row_graph(row: Mapping[str, Any], *, source_tokenizer_fingerprint: str,
                   tokenizer, expanded_tokenizer_fingerprint: str,
                   context_weight: float = 0.05) -> dict:
    """Represent one complete serving-rendered invocation as a one-node graph.

    Prompt/history tokens receive the declared lower context weight, while the
    exact assistant suffix retains full target weight. Existing row token IDs
    are valid because the expanded tokenizer preserves the pinned source
    vocabulary rows; row decode equality checks that fact per document.
    """
    if not math.isfinite(context_weight) or context_weight < 0:
        raise ValueError("context weight must be finite and nonnegative")
    ids, text, split, groups, identifier = _validate_row(row,
        source_tokenizer_fingerprint=source_tokenizer_fingerprint,
        tokenizer=tokenizer, expanded_tokenizer_fingerprint=expanded_tokenizer_fingerprint)
    boundary = row["supervised_suffix_start"]
    weights = [context_weight] * boundary + [1.] * (len(ids)-boundary)
    weights[0] = 0.
    twin = row.get("source_twin_id") or row.get("source_record_ids") or [identifier]
    twin_id = canonical_json_sha256_hex(sorted(str(value) for value in twin))
    return {
        "schema": SCHEMA,
        "id": "text:" + identifier,
        "split": split,
        "source_groups": groups,
        "tokenizer_fingerprint": expanded_tokenizer_fingerprint,
        "nodes": [{"name": "text", "token_ids": ids, "target_weights": weights}],
        "conversion": {"kind": "static-transformation", "role_interpretation": "ordinary-text",
            "source_row_id": identifier,
            "source_twin_id": twin_id, "source_tokenizer_fingerprint": source_tokenizer_fingerprint,
            "source_text_sha256": _sha_text(text), "source_suffix_start": boundary,
            "context_weight": context_weight,
            "context_attestations": row.get("neuralese_context_attestations", [])},
    }


def _wrap_span(ids: list[int], weights: list[float], span: Mapping[str, Any], *,
               open_id: int, close_id: int, role: str, source_tokenizer,
               required_body_sha256: str | None = None):
    # Native contextual text initialization tokenizes the body as its own
    # segment between two structural role tokens. Rebuild that exact input
    # geometry instead of trusting BPE boundaries from the flattened rendered
    # row, where a token can merge across a body edge or represent literal
    # special-token text.
    if span.get("tokenize_payload_alone") is True:
        container_start, container_stop = span.get("container_start"), span.get("container_stop")
        body, payload_ids = span.get("body_text"), span.get("payload_token_ids")
        if (type(container_start) is not int or type(container_stop) is not int
                or not 0 <= container_start < container_stop <= len(ids)
                or ids[container_start] != open_id or ids[container_stop - 1] != close_id
                or not isinstance(body, str) or not isinstance(payload_ids, list)
                or not payload_ids or any(type(token) is not int or token < 0 for token in payload_ids)):
            raise ValueError("native contextual body needs exact structural role markers and standalone token IDs")
        body_sha = _sha_text(body)
        if required_body_sha256 is not None and body_sha != required_body_sha256:
            raise ValueError("native contextual body differs from the authenticated source payload")
        if source_tokenizer.decode(payload_ids, skip_special_tokens=False) != body:
            raise ValueError("standalone contextual body tokens do not decode to the exact authenticated text")
        start = container_start + 1
        stop = start + len(payload_ids)
        wrapped_ids = ids[:container_start] + [open_id] + payload_ids + [close_id] + ids[container_stop:]
        open_weight = weights[container_start]
        close_weight = weights[container_stop - 1]
        wrapped_weights = (weights[:container_start] + [open_weight] + [open_weight] * len(payload_ids)
                           + [close_weight] + weights[container_stop:])
        producing = [[start, stop]] if role == "produce" else []
        return wrapped_ids, wrapped_weights, start, stop, body, body_sha, producing
    start, stop = span.get("start"), span.get("stop")
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(ids):
        raise ValueError("witnessed role span is outside source token IDs")
    body = source_tokenizer.decode(ids[start:stop], skip_special_tokens=False)
    body_sha = _sha_text(body)
    if required_body_sha256 is not None and body_sha != required_body_sha256:
        raise ValueError("witnessed payload text hash differs from the actual source-token span")
    if span.get("role_wrapped") is True:
        container_start = span.get("container_start")
        container_stop = span.get("container_stop")
        if (type(container_start) is not int or type(container_stop) is not int
                or container_start + 1 != start or stop + 1 != container_stop
                or not 0 <= container_start < start < stop < container_stop <= len(ids)
                or ids[container_start] != open_id or ids[container_stop - 1] != close_id):
            raise ValueError("prewrapped witnessed span does not contain its exact structural role markers")
        producing = [[start, stop]] if role == "produce" else []
        return ids, weights, start, stop, body, body_sha, producing
    container_start = span.get("container_start", start)
    container_stop = span.get("container_stop", stop)
    if (type(container_start) is not int or type(container_stop) is not int
            or not 0 <= container_start <= start < stop <= container_stop <= len(ids)):
        raise ValueError("witnessed role container does not enclose its exact payload token span")
    shifted = container_start + 1
    wrapped_ids = ids[:container_start] + [open_id] + ids[start:stop] + [close_id] + ids[container_stop:]
    wrapped_weights = (weights[:container_start] + [weights[start]] + weights[start:stop]
                       + [weights[stop-1]] + weights[container_stop:])
    producing = [[shifted, shifted + stop-start]] if role == "produce" else []
    return wrapped_ids, wrapped_weights, shifted, shifted + stop-start, body, body_sha, producing


def locate_attested_role_span(row: Mapping[str, Any], *, tokenizer, payload_sha256: str,
                              open_marker: str = "<|neuralese|>",
                              close_marker: str = "<|/neuralese|>") -> dict:
    """Locate one exact attested payload inside explicit legacy marker syntax.

    Marker-like strings in prompts are not roles: a span is returned only when
    its enclosed text hashes to the supplied authenticated payload digest and
    token offsets reproduce the source row's exact IDs. Ambiguity is an error.
    """
    text, ids = row.get("text"), row.get("token_ids")
    if not isinstance(text, str) or not isinstance(ids, list):
        raise ValueError("attested role location needs a rendered text row")
    pattern = re.compile(re.escape(open_marker) + r"(.*?)" + re.escape(close_marker), re.DOTALL)
    matches = [match for match in pattern.finditer(text)
               if _sha_text(match.group(1)) == payload_sha256]
    if len(matches) != 1:
        raise ValueError(f"attested role payload has {len(matches)} exact marker matches; expected one")
    match = matches[0]
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except TypeError as exc:
        raise ValueError("tokenizer must provide exact offset mappings for role-boundary verification") from exc
    encoded_ids = encoded["input_ids"]
    if encoded_ids and isinstance(encoded_ids[0], list):
        encoded_ids = encoded_ids[0]
    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    if list(encoded_ids) != ids or len(offsets) != len(ids):
        raise ValueError("whole-string tokenization differs from the authenticated rendered row")
    payload_start, payload_stop = match.start(1), match.end(1)
    container_start, container_stop = match.start(), match.end()

    def exact_range(start, stop):
        selected = [index for index, (left, right) in enumerate(offsets)
                    if right > left and left >= start and right <= stop]
        if not selected or selected != list(range(selected[0], selected[-1]+1)):
            raise ValueError("attested character span is not a contiguous token range")
        if offsets[selected[0]][0] != start or offsets[selected[-1]][1] != stop:
            raise ValueError("attested character boundary splits a token")
        if tokenizer.decode(ids[selected[0]:selected[-1]+1], skip_special_tokens=False) != text[start:stop]:
            raise ValueError("attested token span does not decode to exact source payload")
        return selected[0], selected[-1]+1

    start, stop = exact_range(payload_start, payload_stop)
    cs, ce = exact_range(container_start, container_stop)
    return {"container_start": cs, "start": start, "stop": stop,
            "container_stop": ce, "text": match.group(1),
            "text_sha256": payload_sha256}


def witnessed_pair_graph(writer_row: Mapping[str, Any], reader_row: Mapping[str, Any], *,
                         source_tokenizer, source_tokenizer_fingerprint: str,
                         expanded_tokenizer_fingerprint: str, role_ids: tuple[int, int, int, int],
                         writer_span: Mapping[str, Any], reader_span: Mapping[str, Any],
                         payload_sha256: str, source_span_id: str, edge_id: str, chain_id: str,
                         context_weight: float = 0.05) -> dict:
    """Build a two-invocation static producer/consumer graph from an exact edge.

    The caller supplies spans only after authenticating the writer/reader
    receipts, payload text, source groups and chain identity. This verifier
    checks exact visible payload bytes against both tokenized rows. Role marker
    IDs are the four tokens provisioned on the destination model.
    """
    if writer_row.get("split") != reader_row.get("split"):
        raise ValueError("native producer and consumer must share one explicit split")
    if not isinstance(chain_id, str) or not chain_id:
        raise ValueError("producer/consumer edge needs its authenticated whole-trajectory identity")
    if (len(role_ids) != 4 or any(type(token) is not int or token < 0 for token in role_ids)
            or len(set(role_ids)) != 4):
        raise ValueError("native graph needs four distinct provisioned role-token IDs")
    if not isinstance(source_span_id, str) or not source_span_id or not isinstance(edge_id, str) or not edge_id:
        raise ValueError("producer/consumer graph needs exact span and edge identities")
    writer = _validate_row(writer_row, source_tokenizer_fingerprint=source_tokenizer_fingerprint,
        tokenizer=source_tokenizer, expanded_tokenizer_fingerprint=expanded_tokenizer_fingerprint)
    reader = _validate_row(reader_row, source_tokenizer_fingerprint=source_tokenizer_fingerprint,
        tokenizer=source_tokenizer, expanded_tokenizer_fingerprint=expanded_tokenizer_fingerprint)
    wids, wtext, wsplit, wgroups, wid = writer
    rids, rtext, rsplit, rgroups, rid = reader
    if not math.isfinite(context_weight) or context_weight < 0:
        raise ValueError("context weight must be finite and nonnegative")
    wboundary, rboundary = writer_row["supervised_suffix_start"], reader_row["supervised_suffix_start"]
    wweights = [context_weight] * wboundary + [1.] * (len(wids)-wboundary)
    rweights = [context_weight] * rboundary + [1.] * (len(rids)-rboundary)
    wweights[0] = rweights[0] = 0.
    p_open, p_close, c_open, c_close = role_ids
    wids, wweights, wstart, wstop, body, body_sha, producing = _wrap_span(
        wids, wweights, writer_span, open_id=p_open, close_id=p_close, role="produce",
        source_tokenizer=source_tokenizer, required_body_sha256=payload_sha256)
    rids, rweights, rstart, rstop, read_body, read_sha, _ = _wrap_span(
        rids, rweights, reader_span, open_id=c_open, close_id=c_close, role="consume",
        source_tokenizer=source_tokenizer, required_body_sha256=payload_sha256)
    if read_body != body or read_sha != body_sha:
        raise ValueError("consumer span text is not the exact producer payload")
    # The supplied imported body is teacher-forced state context, not a
    # consumer-authored output target. Preserve loss on the role markers and
    # on all later consumer response tokens.
    rweights[rstart:rstop] = [0.] * (rstop-rstart)
    # Inject imported state at the first token of the consumer payload, after
    # its consume-open marker. The own text remains present as aligned context.
    nodes = [
        {"name": "producer", "token_ids": wids, "target_weights": wweights,
         "producing": producing},
        {"name": "consumer", "token_ids": rids, "target_weights": rweights,
         "imports": [{"at": rstart, "source": "producer", "start": wstart, "stop": wstop}]},
    ]
    source_ids = sorted({wid, rid})
    pair_id = canonical_json_sha256_hex({"edge": edge_id, "source_ids": source_ids,
        "source_span_id": source_span_id, "payload_sha256": payload_sha256,
        "chain_id": chain_id})
    return {"schema": SCHEMA, "id": "edge:" + edge_id, "split": wsplit,
        "source_groups": sorted(set(wgroups) | set(rgroups)),
        "tokenizer_fingerprint": expanded_tokenizer_fingerprint, "nodes": nodes,
        "source_pairing": {"pair_id": pair_id, "source_span_id": source_span_id,
            "edge_id": edge_id, "chain_id": chain_id, "source_ids": source_ids,
            "payloads": [{"invocation": "producer", "start": wstart, "stop": wstop,
                "text": body, "source_span_id": source_span_id,
                "text_sha256": body_sha, "coordinates": "own-token-rows"}]}}
