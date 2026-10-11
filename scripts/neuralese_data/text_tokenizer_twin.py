#!/usr/bin/env python3
"""Tokenizer twin of a rendered native-gold text file (natlang.native_gold_chat/3).

Every row of the source text was produced by ``native_gold_packet(tokenizer,
messages, target, tools)``; everything upstream of that call (record → crisp
messages, context hydration, dedup grouping) is tokenizer-independent. This
script recovers ``(messages, target, tools)`` from each source row's own token
IDs, proves the recovery exact by re-rendering it with the source tokenizer
through the pinned renderer snapshot (text, token IDs and supervised boundary
must be byte-identical), then renders the same call with the twin tokenizer.

Document identity (id, split, source_groups, source_record_ids, attestation
lists) is copied from the source row; the order is the source order. A row
whose recovery does not round-trip, or that the twin template cannot render,
is written to ``omissions.jsonl`` with the reason; nothing is dropped silently.

Usage (CPU, inside the natlang-neuralese image):
  text_tokenizer_twin.py --source-text T --source-tokenizer DIR --twin-tokenizer DIR \
      --renderer-package-root SNAPSHOT --out-dir OUT [--workers N]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import keyword
import re
import sys
from multiprocessing import Pool
from pathlib import Path

SCHEMA = "natlang.text-tokenizer-twin/1"
_G: dict = {}


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _init(src_tok, twin_tok, package, reasoning=None):
    sys.path.insert(0, str(package))
    from transformers import AutoTokenizer
    from natlang_neuralese.data import text_corpus
    _G["tc"] = text_corpus
    _G["src"] = AutoTokenizer.from_pretrained(src_tok, local_files_only=True, trust_remote_code=False)
    _G["twin"] = AutoTokenizer.from_pretrained(twin_tok, local_files_only=True, trust_remote_code=False)
    src = _G["src"]
    # Structure tokens: every added token the template can emit as one ID. Content spellings of the
    # renderer's escaped specials are tokenized split, so an unsplit ID here is template structure.
    structure = {src.bos_token, "<|im_start|>", "<|im_end|>", "<|tool_call_start|>", "<|tool_call_end|>",
                 "<think>", "</think>"}
    _G["special_ids"] = {i: tok.content for i, tok in src.added_tokens_decoder.items() if tok.content in structure}
    _G["reasoning"] = reasoning


def _items(ids):
    """Token IDs → [(kind, string)]: 'S' for an unsplit special token, 'T' for decoded ordinary runs."""
    src, specials, out, run = _G["src"], _G["special_ids"], [], []
    for i in ids:
        if i in specials:
            if run:
                out.append(("T", src.decode(run, skip_special_tokens=False, clean_up_tokenization_spaces=False)))
                run = []
            out.append(("S", specials[i]))
        else:
            run.append(i)
    if run:
        out.append(("T", src.decode(run, skip_special_tokens=False, clean_up_tokenization_spaces=False)))
    return out


class _JsonNames(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id in ("true", "false", "null"):
            return ast.copy_location(ast.Constant({"true": True, "false": False, "null": None}[node.id]), node)
        raise ValueError(f"unexpected name in tool call: {node.id}")


def _literal(node):
    return ast.literal_eval(_JsonNames().visit(node))


_KW = re.compile(r"([(]|, )(" + "|".join(keyword.kwlist) + r")=")


def _parse_calls(text: str) -> list:
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError:  # an argument named like a Python keyword, e.g. eval(continue=True)
        tree = ast.parse(_KW.sub(lambda m: m.group(1) + "__kwarg_" + m.group(2) + "=", text), mode="eval").body
    if not isinstance(tree, ast.List):
        raise ValueError("tool-call block is not a list")
    calls = []
    for call in tree.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.args:
            raise ValueError("tool call is not name(keyword=value, ...)")
        calls.append({"type": "function", "function": {
            "name": call.func.id,
            "arguments": {kw.arg.removeprefix("__kwarg_") if kw.arg.startswith("__kwarg_")
                          and keyword.iskeyword(kw.arg[8:]) else kw.arg: _literal(kw.value)
                          for kw in call.keywords}}})
    return calls


def _until(items, k, stops):
    """Join item strings from k up to (not including) the first item in `stops` → (text, index)."""
    parts = []
    while k < len(items) and items[k] not in stops:
        parts.append(items[k][1])
        k += 1
    if k == len(items):
        raise ValueError(f"missing {sorted(stops)}")
    return "".join(parts), k


_END, _CALL, _CALL_END = ("S", "<|im_end|>"), ("S", "<|tool_call_start|>"), ("S", "<|tool_call_end|>")


def _assistant_body(items, k):
    """Items of an assistant turn from k (just after 'assistant\n') → (message, index after '<|im_end|>')."""
    message = {"role": "assistant"}
    if items[k] == ("S", "<think>"):
        message["reasoning_content"], k = _until(items, k + 1, {("S", "</think>")})
        k += 1
    message["content"], k = _until(items, k, {_END, _CALL})
    if items[k] == _CALL:
        body, k = _until(items, k + 1, {_CALL_END})
        message["tool_calls"] = _parse_calls(body)
        k += 1
    if items[k] != _END:
        raise ValueError(f"unexpected assistant tail: {items[k:k + 3]}")
    return message, k + 1


def _split_tools(system: str):
    marker = "List of tools: ["
    start = 0
    while True:
        at = system.find(marker, start)
        if at < 0:
            return system, []
        head = system[:at]
        if head and not head.endswith("\n"):
            start = at + 1
            continue
        body, tools, dec, pos = system[at + len(marker):], [], json.JSONDecoder(), 0
        try:
            while True:
                tool, pos = dec.raw_decode(body, pos)
                tools.append(tool)
                if body[pos:pos + 2] == ", ":
                    pos += 2
                    continue
                break
        except json.JSONDecodeError:
            start = at + 1
            continue
        if body[pos:] == "]":
            return (head[:-1] if head else head), tools
        start = at + 1


def _header(items, k):
    """'<|im_start|>role\n' at k → (role, items with the header text removed, index of the body)."""
    if items[k] != ("S", "<|im_start|>") or items[k + 1][0] != "T":
        raise ValueError(f"expected a turn start at item {k}")
    role, nl, first = items[k + 1][1].partition("\n")
    if not nl:
        raise ValueError("turn header lacks newline")
    items = items[:k + 1] + ([("T", first)] if first else []) + items[k + 2:]
    return role, items, k + 1


def recover(row):
    """Source row → (messages, target, tools) for native_gold_packet."""
    ids, boundary = row["token_ids"], row["supervised_suffix_start"]
    items, reply = _items(ids[:boundary]), _items(ids[boundary:])
    if "".join(s for _, s in items + reply) != row["text"]:
        raise ValueError("decoded token runs do not reproduce the text")
    if not items or items[0] != ("S", _G["src"].bos_token):
        raise ValueError("missing BOS")
    if items[-2:] != [("S", "<|im_start|>"), ("T", "assistant\n")]:
        raise ValueError("prompt does not end with the generation prompt")
    messages, tools = [], []
    body, k = items[1:-2], 0
    while k < len(body):
        role, body, k = _header(body, k)
        if role == "assistant":
            message, k = _assistant_body(body, k)
        else:
            content, k = _until(body, k, {_END})
            k += 1
            if role == "system" and not messages and not tools:
                content, tools = _split_tools(content)
                if not content:
                    message = None
            message = {"role": role, "content": content} if content or role != "system" else None
        if k < len(body) and body[k] == ("T", "\n"):
            k += 1
        elif k < len(body):
            raise ValueError(f"turn not followed by a newline at item {k}")
        if message is not None:
            messages.append(message)
    target, k = _assistant_body(reply, 0)
    if reply[k:] != [("T", "\n")]:
        raise ValueError("reply does not end with '<|im_end|>\\n'")
    return messages, target, tools


def work(args):
    index, line = args
    row = json.loads(line)
    tc = _G["tc"]
    ident = {k: row[k] for k in ("id", "split", "source_groups", "source_record_ids")}
    try:
        messages, target, tools = recover(row)
    except (ValueError, IndexError, SyntaxError, KeyError) as exc:
        return index, None, {**ident, "reason": "source_row_recovery_failed", "detail": str(exc)[:300]}, None
    # The source template renders only the last history assistant's reasoning; the twin template may
    # render more. Restore every history assistant's reasoning_content from the source native record,
    # matched on the reasoning the source text does show.
    asst = [m for m in messages if m["role"] == "assistant"]
    shown = asst[-1].get("reasoning_content") if asst else None
    cands = {t for t in (_G["reasoning"] or {}).get(row["id"], ())
             if len(t) == len(asst) and (not asst or (t[-1] or None) == shown)}
    if len(cands) != 1:
        reason = "no_matching_native_record" if not cands else "ambiguous_native_record_reasoning"
        return index, None, {**ident, "reason": reason, "detail": f"{len(cands)} candidates"}, None
    restored = 0
    for m, r in zip(asst, next(iter(cands))):
        if isinstance(r, str) and r.startswith("\0non-string:"):
            return index, None, {**ident, "reason": "non_string_history_reasoning"}, None
        if isinstance(r, str) and r and m.get("reasoning_content") != r:
            m["reasoning_content"] = r
            restored += 1
    try:
        text, ids, boundary = tc.native_gold_packet(_G["src"], messages, target, tools)
    except Exception as exc:  # noqa: BLE001 - recorded as an explicit omission
        return index, None, {**ident, "reason": "source_rerender_failed", "detail": str(exc)[:300]}, None
    if (text, ids, boundary) != (row["text"], row["token_ids"], row["supervised_suffix_start"]):
        what = [n for n, a, b in (("text", text, row["text"]), ("token_ids", ids, row["token_ids"]),
                                  ("boundary", boundary, row["supervised_suffix_start"])) if a != b]
        return index, None, {**ident, "reason": "source_roundtrip_mismatch", "detail": ",".join(what)}, None
    try:
        t_text, t_ids, t_boundary = tc.native_gold_packet(_G["twin"], messages, target, tools)
    except Exception as exc:  # noqa: BLE001
        return index, None, {**ident, "reason": "twin_render_failed", "detail": str(exc)[:300]}, None
    out = dict(row)
    out.update(text=t_text, token_ids=t_ids, supervised_suffix_start=t_boundary,
               tokenizer_sha256=tc.tokenizer_fingerprint(_G["twin"]))
    history_think = sum(1 for m in messages if m["role"] == "assistant" and "<think>" in (m.get("content") or ""))
    stats = {"restored_history_reasoning": restored, "history_assistant_turns": len(asst), "tokens": len(t_ids), "boundary": t_boundary, "source_tokens": len(row["token_ids"]),
             "history_think_messages": history_think, "twin_think_spans": t_text.count("<think>"),
             "source_think_spans": row["text"].count("<think>")}
    return index, out, None, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-text", type=Path, required=True)
    ap.add_argument("--source-tokenizer", required=True)
    ap.add_argument("--twin-tokenizer", required=True)
    ap.add_argument("--renderer-package-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--native-records", type=Path, nargs="+", required=True,
                    help="native record files holding the source rows' records (history reasoning_content)")
    args = ap.parse_args()
    out = args.out_dir
    if out.exists() and any(p.name != "text-data.jsonl" or p.stat().st_size for p in out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    needed = set()
    with args.source_text.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                needed.add(json.loads(line)["id"])
    reasoning, record_files = {}, []
    for path in args.native_records:
        record_files.append({"path": str(path), "sha256": file_sha(path)})
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("id") in needed and isinstance(rec.get("messages"), list):
                    reasoning.setdefault(rec["id"], set()).add(tuple(
                        r if r is None or isinstance(r, str) else "\0non-string:" + json.dumps(r, sort_keys=True)
                        for r in (m.get("reasoning_content") for m in rec["messages"]
                                  if m.get("role") == "assistant")))
    _init(args.source_tokenizer, args.twin_tokenizer, args.renderer_package_root.resolve(), reasoning)
    with args.source_text.open(encoding="utf-8") as f:
        lines = [(i, line) for i, line in enumerate(f) if line.strip()]
    if args.limit:
        lines = lines[:args.limit]
    results = [None] * len(lines)
    with Pool(args.workers, initializer=_init,
              initargs=(args.source_tokenizer, args.twin_tokenizer, args.renderer_package_root.resolve(), reasoning)) as pool:
        for index, row, omission, stats in pool.imap_unordered(work, lines, chunksize=4):
            results[index] = (row, omission, stats)
    fields = ("id", "split", "source_groups", "source_record_ids")
    with (out / "text-data.jsonl").open("w", encoding="utf-8") as text_f, \
            (out / "omissions.jsonl").open("w", encoding="utf-8") as om_f, \
            (out / "row-stats.jsonl").open("w", encoding="utf-8") as st_f:
        for (index, line), (row, omission, stats) in zip(lines, results):
            if omission is not None:
                om_f.write(json.dumps({"source_line": index, **omission}, ensure_ascii=False, sort_keys=True) + "\n")
                continue
            text_f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            st_f.write(json.dumps({"id": row["id"], "split": row["split"], **stats}, sort_keys=True) + "\n")
    # Per-row identity check against the source, in order.
    src_ident = [{k: json.loads(line)[k] for k in fields} for _, line in lines]
    twin_ident = []
    with (out / "text-data.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            twin_ident.append({k: r[k] for k in fields})
    omitted = sum(1 for r in results if r[1] is not None)
    mismatches = [i for i, (a, b) in enumerate(zip(src_ident, twin_ident)) if a != b]
    tc = _G["tc"]
    summary = {
        "schema": SCHEMA,
        "source_text": {"path": str(args.source_text), "sha256": file_sha(args.source_text), "rows": len(lines)},
        "twin_text": {"path": "text-data.jsonl", "sha256": file_sha(out / "text-data.jsonl"), "rows": len(twin_ident)},
        "source_tokenizer": {"path": args.source_tokenizer, "fingerprint": tc.tokenizer_fingerprint(_G["src"])},
        "twin_tokenizer": {"path": args.twin_tokenizer, "fingerprint": tc.tokenizer_fingerprint(_G["twin"])},
        "renderer": {"package_root": str(args.renderer_package_root),
                     "text_corpus_sha256": file_sha(args.renderer_package_root / "natlang_neuralese/data/text_corpus.py"),
                     "chat_sha256": file_sha(args.renderer_package_root / "natlang_neuralese/serve/chat.py"),
                     "function": "natlang_neuralese.data.text_corpus.native_gold_packet",
                     "rendering": "natlang.native_gold_chat/3"},
        "identity": {"fields": list(fields), "compared_rows": min(len(src_ident), len(twin_ident)),
                     "per_row_equal_in_order": omitted == 0 and not mismatches and len(src_ident) == len(twin_ident),
                     "mismatched_rows": mismatches[:50]},
        "native_record_files": record_files,
        "omitted_rows": omitted,
        "source_roundtrip_exact_rows": len(twin_ident),
    }
    (out / "twin-identity-check.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
