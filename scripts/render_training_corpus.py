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
import shutil
import sys
from pathlib import Path
from typing import Any

CHUNK_ROWS = 128
RENDERER_VERSION = "transformers-chat-template/4"


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
                                        'curriculum_lane', 'training_track', 'difficulty', 'syntax_checked') if key in row},
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
    template = getattr(tokenizer, 'chat_template', '')
    # Templates that forget earlier reasoning otherwise rewrite the already rendered prompt
    # when a new assistant target is appended. Use the same policy when serving.
    template_kwargs = {'preserve_thinking': True} if isinstance(template, str) and 'preserve_thinking' in template else {}
    rendered = tokenizer.apply_chat_template(template_messages, tools=tools or None, tokenize=False,
                                               add_generation_prompt=add_generation_prompt, **template_kwargs)
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
    # Render the real closed assistant turn. A synthetic following user query
    # can make templates drop this target's reasoning or rewrite history.
    complete = _call_template(tokenizer, messages + [target], tools, False)
    if not complete.startswith(prompt):
        raise ValueError(f"{turn.get('id', '<unknown>')}: tokenizer changed the assistant prefix")
    suffix = complete[len(prompt):]
    end = suffix.find(end_token)
    if end < 0:
        raise ValueError(f"{turn.get('id', '<unknown>')}: assistant end token is absent from rendered target")
    if turn.get("teacher_reasoning") and turn["teacher_reasoning"] not in suffix[:end]:
        raise ValueError(f"{turn.get('id', '<unknown>')}: template dropped teacher reasoning")
    result = {key: turn[key] for key in (
        "id", "program_id", "source_groups", "split", "family", "task_family", "task_kind", "task_modality", "skill", "quality",
        "training_admission", "source", "license", "source_ids", "source_revisions",
        "teacher_trajectory_id", "teacher_trajectory_digest", "execution_verified",
        "implementation_sha256", "behavioral_evidence", "evidence", "generation", "verification",
        "curriculum_lane", "training_track", "difficulty", "syntax_checked", "gold_sources") if key in turn}
    if turn.get('provenance', {}).get('source_conversion'):
        result['source_conversion'] = turn['provenance']['source_conversion']
    result.update(renderer="transformers-chat-template", context_items=len(messages),
                  prompt=prompt, completion=suffix[:end + len(end_token)])
    if turn.get('teacher_reasoning') and turn.get('teacher_reasoning_trained') is False:
        reasoning_start = result['completion'].find(turn['teacher_reasoning'])
        boundary = result['completion'].find('</think>', reasoning_start + len(turn['teacher_reasoning']))
        if boundary < 0:
            raise ValueError('untrained reasoning needs an explicit template reasoning boundary')
        result['completion_masked'] = boundary + len('</think>')
    return result


def _assistant_end_token(tokenizer: Any, requested: str | None = None) -> str:
    """Resolve the terminator the tokenizer's closed assistant template emits.

    Some tokenizers expose a generic EOS while their chat format uses another
    special token (MiniCPM's </s> versus <|im_end|>, for example).
    """
    default = getattr(tokenizer, "eos_token", None)
    special = list(getattr(tokenizer, "all_special_tokens", None) or [])
    # Some models register their chat controls as added tokens rather than
    # setting the tokenizer's special=True flag. The closed template still
    # supplies the authoritative assistant boundary.
    added = getattr(tokenizer, "get_added_vocab", None)
    if callable(added):
        special.extend(added().keys())
    if special and callable(getattr(tokenizer, "apply_chat_template", None)):
        closed = _call_template(tokenizer, [
            {"role": "user", "content": "Check the assistant message format."},
            {"role": "assistant", "content": "Format check complete."},
        ], [], False).rstrip()
        endings = [token for token in special if isinstance(token, str) and token and closed.endswith(token)]
        if not endings:
            raise ValueError("closed assistant template must end with a registered tokenizer token")
        inferred = max(endings, key=len)
        if requested is not None and requested != inferred:
            raise ValueError("requested end token differs from the closed assistant template")
        return inferred
    token = requested if requested is not None else default
    if default is not None and requested is not None and requested != default:
        raise ValueError("requested end token is not verifiable against this tokenizer")
    if not isinstance(token, str) or not token:
        raise ValueError("--end-token or tokenizer.eos_token must be a nonempty string")
    return token


def _tokenizer_info(tokenizer: Any, model: str, revision: str | None,
                    *, end_token: str | None = None) -> tuple[str, dict[str, Any]]:
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
                "template_sha256": _sha(template.encode()), "end_token": _assistant_end_token(tokenizer, end_token),
                "template_kwargs": {"preserve_thinking": True} if "preserve_thinking" in template else {}}
    return template, renderer


def render_corpus(inputs: list[Path], output: Path, *, model: str, revision: str | None = None,
                  end_token: str | None = None, chunk_rows: int = CHUNK_ROWS,
                  tokenizer: Any | None = None, should_stop=lambda: False, streaming: bool = False) -> int:
    if revision is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("--revision must be an immutable 40-character Hugging Face commit SHA")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if tokenizer is None:
        from transformers import AutoTokenizer  # Imported only for an actual render.
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    _, renderer = _tokenizer_info(tokenizer, model, revision, end_token=end_token)
    end_token = renderer["end_token"]
    if streaming:
        return _render_corpus_streaming(inputs, output, renderer=renderer, tokenizer=tokenizer,
                                        end_token=end_token, chunk_rows=chunk_rows, should_stop=should_stop)
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
    def require_rendered_decisions():
        if not committed and any(item['reason'] == 'invalid_training_view' for item in rejections):
            raise ValueError(f"no training decisions could be rendered; inspect {rejection_path}")
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
            require_rendered_decisions()
            return 0
    else:
        _atomic_new(output, data)
    if prior_manifest is None:
        _atomic_new(final_manifest_path, (json.dumps(final_manifest, indent=2) + "\n").encode())
    require_rendered_decisions()
    return 0


def _iter_input_chunks(paths: list[Path], chunk_rows: int):
    chunk = []
    for path in paths:
        resolved = path.resolve()
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
                chunk.append(row)
                if len(chunk) == chunk_rows:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def _stream_hash(path: Path) -> str:
    return _file_sha(path)


def _render_corpus_streaming(inputs: list[Path], output: Path, *, renderer: dict[str, Any], tokenizer: Any,
                             end_token: str, chunk_rows: int, should_stop=lambda: False) -> int:
    """Render with bounded input/output memory while retaining resumable chunk commits."""
    input_identity = [{"path": str(p.resolve()), "sha256": _file_sha(p.resolve())} for p in inputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    input_bytes = sum(Path(p).stat().st_size for p in inputs)
    required_bytes = input_bytes * 3 + 1024 ** 3
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < required_bytes:
        raise OSError(f"streaming render requires about {required_bytes} free bytes "
                      f"(3x {input_bytes} input bytes plus 1 GiB reserve); only {free_bytes} available")
    identity = {"renderer": renderer, "inputs": input_identity, "chunk_rows": chunk_rows,
                "renderer_sha256": _file_sha(Path(__file__)), "streaming": "jsonl-chunks/1"}
    identity_sha = _sha(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    cache = output.with_name(output.name + ".cache")
    cache.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache / "manifest.json"
    manifest = {"version": 2, "identity": identity, "identity_sha256": identity_sha, "chunks": []}
    if cache_manifest_path.exists():
        manifest = json.loads(cache_manifest_path.read_text())
        if manifest.get("version") != 2 or manifest.get("identity_sha256") != identity_sha:
            raise ValueError("existing streaming shard cache identity differs from renderer or inputs")

    chunks = iter(_iter_input_chunks(inputs, chunk_rows))
    cursor = 0
    for index, chunk in enumerate(chunks):
        if index < len(manifest["chunks"]):
            committed = manifest["chunks"][index]
            if (committed.get("index") != index or committed.get("input_row_start") != cursor or
                    committed.get("input_rows") != len(chunk) or
                    committed.get("input_sha256") != _sha(_jsonl_bytes(chunk))):
                raise ValueError(f"committed chunk input mismatch at index {index}")
            data_path, reject_path = cache / committed["file"], cache / committed["rejections_file"]
            if _file_sha(data_path) != committed.get("sha256") or _file_sha(reject_path) != committed.get("rejections_sha256"):
                raise ValueError(f"committed chunk hash mismatch at index {index}")
            cursor += len(chunk)
            continue
        selected, rejected = [], []
        for original in chunk:
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
        data_name, reject_name = f"stream-{index:08d}.jsonl", f"stream-{index:08d}.rejected.jsonl"
        payload, rejects = _jsonl_bytes(selected), _jsonl_bytes(rejected)
        _atomic(cache / data_name, payload)
        _atomic(cache / reject_name, rejects)
        item = {"index": index, "input_row_start": cursor, "input_rows": len(chunk),
                "input_sha256": _sha(_jsonl_bytes(chunk)), "file": data_name, "sha256": _sha(payload),
                "rows": len(selected), "rejections_file": reject_name,
                "rejections_sha256": _sha(rejects), "rejections": dict(Counter(r['reason'] for r in rejected))}
        manifest["chunks"].append(item)
        _atomic(cache_manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
        cursor += len(chunk)
        if should_stop():
            return 75
    if cursor == 0 and manifest["chunks"]:
        raise ValueError("cache has chunks but current input is empty")
    if len(manifest["chunks"]) != sum(1 for _ in _iter_input_chunks(inputs, chunk_rows)):
        raise ValueError("streaming shard cache is incomplete")
    if any(_file_sha(Path(item['path'])) != item['sha256'] for item in input_identity):
        raise ValueError('source inputs changed during rendering')

    rejection_path = output.with_name(output.name + '.rejected.jsonl')
    rejection_tmp = rejection_path.with_name(rejection_path.name + f".tmp-{os.getpid()}")
    rejection_hash = hashlib.sha256(); rejection_counts = Counter()
    with rejection_tmp.open("wb") as out:
        for item in manifest["chunks"]:
            part = cache / item["rejections_file"]
            with part.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    out.write(block); rejection_hash.update(block)
            rejection_counts.update(item.get("rejections", {}))
        out.flush(); os.fsync(out.fileno())
    rejection_sha = rejection_hash.hexdigest()
    if rejection_path.exists():
        if _file_sha(rejection_path) != rejection_sha:
            rejection_tmp.unlink(missing_ok=True)
            raise ValueError('existing rendering rejection ledger differs from streaming cache')
        rejection_tmp.unlink()
    else:
        rejection_tmp.replace(rejection_path)
    rows_written = sum(item["rows"] for item in manifest["chunks"])

    output_tmp = output.with_name(output.name + f".tmp-{os.getpid()}")
    output_hash = hashlib.sha256()
    with output_tmp.open("wb") as out:
        for item in manifest["chunks"]:
            with (cache / item["file"]).open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    out.write(block); output_hash.update(block)
        out.flush(); os.fsync(out.fileno())
    output_sha = output_hash.hexdigest()
    final_manifest = {"version": "natlang.sft.native/1", "source": [str(p.resolve()) for p in inputs],
                      "source_sha256": [_file_sha(p.resolve()) for p in inputs], "rows": rows_written,
                      "renderer": renderer, "identity_sha256": identity_sha, "sha256": output_sha,
                      "rejections": dict(rejection_counts), "rejections_sha256": rejection_sha}
    final_manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() and _file_sha(output) != output_sha:
        output_tmp.unlink(missing_ok=True)
        raise ValueError(f"existing output differs from verified streaming shard cache: {output}")
    if not output.exists():
        output_tmp.replace(output)
    else:
        output_tmp.unlink()
    if final_manifest_path.exists():
        old = json.loads(final_manifest_path.read_text())
        if old.get("identity_sha256") != identity_sha or old.get("sha256") != output_sha:
            raise ValueError(f"existing output manifest differs: {final_manifest_path}")
    else:
        _atomic_new(final_manifest_path, (json.dumps(final_manifest, indent=2) + "\n").encode())
    if rows_written == 0 and rejection_counts.get("invalid_training_view", 0):
        raise ValueError(f"no training decisions could be rendered; inspect {rejection_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", help="immutable Hugging Face commit SHA")
    parser.add_argument("--end-token", help="optional; otherwise inferred from the tokenizer's closed assistant template, then eos_token")
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--streaming", action="store_true", help="stream inputs and committed shards to bound memory")
    args = parser.parse_args(argv)
    stopping = False
    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    old_int, old_term = signal.signal(signal.SIGINT, request_stop), signal.signal(signal.SIGTERM, request_stop)
    try:
        return render_corpus(args.inputs, args.output, model=args.model, revision=args.revision,
                             end_token=args.end_token, chunk_rows=args.chunk_rows,
                             should_stop=lambda: stopping, streaming=args.streaming)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
