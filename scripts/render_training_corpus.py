#!/usr/bin/env python3
"""Render standard NatLang turns or assembled code_sft rows with a local HF tokenizer."""
from __future__ import annotations

import argparse
from collections import Counter
import fnmatch
import hashlib
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any

CHUNK_ROWS = 128
RENDERER_VERSION = "transformers-chat-template/1"


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + f".tmp-{os.getpid()}")
    with staged.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    staged.replace(path)


def _atomic_new(path: Path, data: bytes) -> None:
    """Install a completed output without replacing anything that appeared meanwhile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + f".tmp-{os.getpid()}")
    with staged.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def _read_inputs(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows, identities = [], []
    for path in paths:
        resolved = path.resolve()
        identities.append({"path": str(resolved), "sha256": _file_sha(resolved)})
        with resolved.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{resolved}:{number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{resolved}:{number}: expected a JSON object")
                rows.append(row)
    return rows, identities


def _as_turn(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("kind") == "code_sft":
        if row.get("training_admission", {}).get("approved") is False:
            return row
        if not row.get("syntax_checked") or not row.get("completion"):
            raise ValueError(f"{row.get('id', '<unknown>')}: code_sft row must be syntax_checked and have completion")
        return {
            **{key: row[key] for key in ('behavioral_evidence', 'evidence', 'generation', 'verification',
                                        'curriculum_lane', 'difficulty', 'syntax_checked') if key in row},
            "id": row.get("id"), "program_id": row.get("program_id"),
            "source_groups": row.get("source_groups", []), "family": row.get("family", "code_corpus"),
            "skill": row.get("skill", "code_generation"), "split": row.get("split"),
            "source": row.get("source"), "quality": row.get("quality"),
            "execution_verified": row.get("execution_verified", False),
            "implementation_sha256": row.get("implementation_sha256"),
            "messages": [{"role": "user", "content": row["prompt"]}], "tools": [],
            "target": {"role": "assistant", "content": row["completion"]},
            "training_admission": {"approved": True, "reason": "syntax-checked code view"},
        }
    return row


def _call_template(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                   add_generation_prompt: bool) -> str:
    template_messages = []
    for message in messages:
        normalized = dict(message)
        if normalized.get('reasoning_content') and not normalized.get('thinking'):
            normalized['thinking'] = normalized['reasoning_content']
        if isinstance(message.get('tool_calls'), list):
            calls = []
            for call in message['tool_calls']:
                normalized_call = dict(call)
                if isinstance(call.get('function'), dict):
                    function = dict(call['function'])
                    if isinstance(function.get('arguments'), str):
                        try:
                            function['arguments'] = json.loads(function['arguments'])
                        except json.JSONDecodeError as exc:
                            raise ValueError('tool call arguments are not valid JSON') from exc
                    if not isinstance(function.get('arguments'), dict):
                        raise ValueError('tool call arguments must be a JSON object')
                    normalized_call['function'] = function
                calls.append(normalized_call)
            normalized['tool_calls'] = calls
        template_messages.append(normalized)
    rendered = tokenizer.apply_chat_template(template_messages, tools=tools or None, tokenize=False,
                                               add_generation_prompt=add_generation_prompt)
    if not isinstance(rendered, str):
        raise ValueError("tokenizer chat template did not return text")
    return rendered


def render_turn(turn: dict[str, Any], tokenizer: Any, end_token: str) -> dict[str, Any] | None:
    if turn.get("training_admission", {}).get("approved") is not True:
        return None
    messages, tools, target = turn.get("messages"), turn.get("tools", []), turn.get("target")
    if not isinstance(messages, list) or not isinstance(tools, list) or not isinstance(target, dict):
        raise ValueError(f"{turn.get('id', '<unknown>')}: missing messages/tools/target training view")
    if target.get('role') != 'assistant':
        raise ValueError('training target must be an assistant turn')
    if end_token in json.dumps(target, ensure_ascii=False) or end_token in str(turn.get('teacher_reasoning', '')):
        raise ValueError('target contains reserved termination token; refusing silent truncation')
    prompt = _call_template(tokenizer, messages, tools, True)
    target = dict(target)
    if turn.get("teacher_reasoning"):
        target["reasoning_content"] = turn["teacher_reasoning"]
    calls = target.get("tool_calls")
    if isinstance(calls, list):
        target["tool_calls"] = [{**call, "id": f"teacher_{i}"} for i, call in enumerate(calls)]
    after = ([{"role": "tool", "tool_call_id": call["id"], "content": "X"} for call in target["tool_calls"]]
             if target.get("tool_calls") else [{"role": "user", "content": "X"}])
    complete = _call_template(tokenizer, messages + [target] + after, tools, False)
    if not complete.startswith(prompt):
        raise ValueError(f"{turn.get('id', '<unknown>')}: tokenizer changed the assistant prefix")
    suffix = complete[len(prompt):]
    end = suffix.find(end_token)
    if end < 0:
        raise ValueError(f"{turn.get('id', '<unknown>')}: assistant end token is absent from rendered target")
    if turn.get("teacher_reasoning") and turn["teacher_reasoning"] not in suffix[:end]:
        raise ValueError(f"{turn.get('id', '<unknown>')}: template dropped teacher reasoning")
    result = {key: turn[key] for key in (
        "id", "program_id", "source_groups", "split", "family", "skill", "quality",
        "training_admission", "source", "license", "source_ids", "source_revisions",
        "teacher_trajectory_id", "teacher_trajectory_digest", "execution_verified",
        "implementation_sha256", "behavioral_evidence", "evidence", "generation", "verification",
        "curriculum_lane", "difficulty", "syntax_checked") if key in turn}
    result.update(renderer="transformers-chat-template", context_items=len(messages),
                  prompt=prompt, completion=suffix[:end + len(end_token)])
    return result


def _tokenizer_info(tokenizer: Any, model: str, revision: str | None) -> tuple[str, dict[str, Any]]:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer must expose one nonempty chat_template string")
    details: dict[str, Any] = {}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        details["backend_json"] = backend.to_str()
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        details["vocab"] = sorted(get_vocab().items())
    details["special_tokens_map"] = getattr(tokenizer, "special_tokens_map", {})
    details["added_vocab"] = getattr(tokenizer, "get_added_vocab", lambda: {})()
    tokenizer_fingerprint = _sha(json.dumps(details, sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":"), default=str).encode())
    local_path = Path(model).expanduser()
    local_artifacts = None
    if local_path.is_dir():
        artifact_names = {
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
            "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "vocab.json", "vocab.txt",
            "merges.txt", "vocab.model", "tekken.json",
        }
        artifact_hashes = []
        for path in sorted(p for p in local_path.rglob("*") if p.is_file() and
                           (p.name in artifact_names or fnmatch.fnmatch(p.name, "vocab.*") or
                            fnmatch.fnmatch(p.name, "merges.*"))):
            artifact_hashes.append({"path": path.relative_to(local_path).as_posix(), "sha256": _file_sha(path)})
        if not artifact_hashes:
            raise ValueError("local model directory has no recognized tokenizer artifacts to fingerprint")
        local_artifacts = _sha(json.dumps(artifact_hashes, sort_keys=True, separators=(",", ":")).encode())
    renderer = {"version": RENDERER_VERSION, "model": model, "revision": revision,
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", model),
                "tokenizer_fingerprint_sha256": tokenizer_fingerprint,
                "local_tokenizer_artifacts_sha256": local_artifacts,
                "template_sha256": _sha(template.encode()), "end_token": tokenizer.eos_token}
    return template, renderer


def render_corpus(inputs: list[Path], output: Path, *, model: str, revision: str | None = None,
                  end_token: str | None = None, chunk_rows: int = CHUNK_ROWS,
                  tokenizer: Any | None = None, should_stop=lambda: False) -> int:
    if revision is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("--revision must be an immutable 40-character Hugging Face commit SHA")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if tokenizer is None:
        from transformers import AutoTokenizer  # Imported only for an actual render.
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    _, renderer = _tokenizer_info(tokenizer, model, revision)
    end_token = end_token or tokenizer.eos_token
    if not isinstance(end_token, str) or not end_token:
        raise ValueError("--end-token or tokenizer.eos_token must be a nonempty string")
    renderer["end_token"] = end_token
    rows, input_identity = _read_inputs(inputs)
    identity = {"renderer": renderer, "inputs": input_identity, "chunk_rows": chunk_rows,
                "renderer_sha256": _file_sha(Path(__file__))}
    identity_sha = _sha(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    cache = output.with_name(output.name + ".cache")
    cache.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache / "manifest.json"
    manifest = {"version": 1, "identity": identity, "identity_sha256": identity_sha, "chunks": []}
    if cache_manifest_path.exists():
        previous = json.loads(cache_manifest_path.read_text())
        if previous.get("identity_sha256") != identity_sha:
            raise ValueError("existing shard cache identity differs from model, inputs, template, or render config")
        manifest = previous
        expected_start = 0
        for chunk_index, chunk in enumerate(manifest["chunks"]):
            if (chunk.get("index") != chunk_index or chunk.get("input_row_start") != expected_start or
                    not isinstance(chunk.get("input_row_end"), int) or chunk["input_row_end"] <= expected_start or
                    chunk["input_row_end"] > len(rows) or chunk.get("input_rows") != chunk["input_row_end"] - expected_start):
                raise ValueError(f"invalid committed chunk boundaries at index {chunk_index}")
            input_chunk = rows[expected_start:chunk["input_row_end"]]
            if chunk.get("input_sha256") != _sha(_jsonl_bytes(input_chunk)):
                raise ValueError(f"committed chunk input hash mismatch: {chunk.get('file')}")
            content = (cache / chunk["file"]).read_bytes()
            if _sha(content) != chunk["sha256"]:
                raise ValueError(f"committed chunk hash mismatch: {chunk['file']}")
            if chunk.get('rejections_sha256') != _sha(_jsonl_bytes(chunk.get('rejections', []))):
                raise ValueError('committed rejection ledger hash mismatch')
            expected_start = chunk["input_row_end"]
        cursor = expected_start
    else:
        cursor = 0
    committed, rejections = [], []
    for start in range(0, len(rows), chunk_rows):
        if start < cursor:
            chunk = next(c for c in manifest["chunks"] if c["input_row_start"] == start)
            committed.extend(json.loads(line) for line in (cache / chunk["file"]).read_text().splitlines())
            rejections.extend(chunk.get('rejections', []))
            continue
        selected, rejected = [], []
        for original in rows[start:start + chunk_rows]:
            try:
                turn = _as_turn(original)
                pair = render_turn(turn, tokenizer, end_token)
            except (ValueError, KeyError) as error:
                rejected.append({'id': original.get('id'), 'source': original.get('source'),
                                 'reason': 'invalid_training_view', 'detail': str(error)})
                continue
            if pair is not None:
                selected.append(pair)
            else:
                rejected.append({'id': original.get('id'), 'source': original.get('source'),
                                 'reason': 'not_explicitly_admitted'})
        index = len(manifest["chunks"])
        name = f"chunk-{index:08d}.jsonl"
        payload = _jsonl_bytes(selected)
        _atomic(cache / name, payload)
        input_end = min(start + chunk_rows, len(rows))
        input_chunk = rows[start:input_end]
        manifest["chunks"].append({"index": index, "input_row_start": start, "input_row_end": input_end,
                                   "input_rows": len(input_chunk), "input_sha256": _sha(_jsonl_bytes(input_chunk)),
                                   "file": name, "sha256": _sha(payload), "rows": len(selected),
                                   'rejections': rejected, 'rejections_sha256': _sha(_jsonl_bytes(rejected))})
        _atomic(cache_manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
        committed.extend(selected)
        rejections.extend(rejected)
        if should_stop():
            return 75
    # All cached and newly rendered chunks are already in input order.
    if cursor < len(rows) and sum(c["input_rows"] for c in manifest["chunks"]) < len(rows):
        raise ValueError("incomplete shard cache")
    data = _jsonl_bytes(committed)
    if any(_file_sha(Path(item['path'])) != item['sha256'] for item in input_identity):
        raise ValueError('source inputs changed during rendering')
    rejection_data = _jsonl_bytes(rejections)
    final_manifest = {"version": "natlang.sft.native/1", "source": [str(p.resolve()) for p in inputs],
                      "source_sha256": [_file_sha(p.resolve()) for p in inputs], "rows": len(committed),
                      "renderer": renderer, "identity_sha256": identity_sha, "sha256": _sha(data),
                      "rejections": dict(Counter(item['reason'] for item in rejections)),
                      'rejections_sha256': _sha(rejection_data)}
    rejection_path = output.with_name(output.name + '.rejected.jsonl')
    if rejection_path.exists():
        if rejection_path.read_bytes() != rejection_data:
            raise ValueError('existing rendering rejection ledger differs from shard cache')
    else:
        _atomic_new(rejection_path, rejection_data)
    final_manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    prior_manifest = None
    if final_manifest_path.exists():
        prior_manifest = json.loads(final_manifest_path.read_text())
        if prior_manifest.get("identity_sha256") != identity_sha:
            raise ValueError(f"existing output manifest identity differs: {final_manifest_path}")
        if prior_manifest.get("sha256") != _sha(data):
            raise ValueError(f"existing output manifest data hash differs: {final_manifest_path}")
    if output.exists():
        existing_data = output.read_bytes()
        if existing_data != data:
            raise ValueError(f"existing output differs from verified shard cache: {output}")
        if prior_manifest is not None:
            if prior_manifest.get("sha256") != _sha(existing_data):
                raise ValueError(f"existing output identity or hash differs: {output}")
            return 0
    else:
        _atomic_new(output, data)
    if prior_manifest is None:
        _atomic_new(final_manifest_path, (json.dumps(final_manifest, indent=2) + "\n").encode())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", help="immutable Hugging Face commit SHA")
    parser.add_argument("--end-token", help="defaults to tokenizer.eos_token")
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    args = parser.parse_args(argv)
    stopping = False
    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    old_int, old_term = signal.signal(signal.SIGINT, request_stop), signal.signal(signal.SIGTERM, request_stop)
    try:
        return render_corpus(args.inputs, args.output, model=args.model, revision=args.revision,
                             end_token=args.end_token, chunk_rows=args.chunk_rows,
                             should_stop=lambda: stopping)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
