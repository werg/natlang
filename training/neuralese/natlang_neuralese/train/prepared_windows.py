"""Persistent cache for tokenized direct-SFT windows.

The cache stores preparation results, not model state. Its identity is made
from the data and the small set of inputs that changes tokenization, masking,
window geometry or source annotations. A miss or a damaged entry is rebuilt
by the caller through the ordinary preparation path.
"""
from __future__ import annotations

import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import sys
import tempfile
from array import array
import fcntl
from contextlib import contextmanager

from ..common import row_index
from ..common.hashing import sha256_file_hex, sha256_file_hex_cached
from ..common.jsonio import canonical_json_bytes
from ..data.text_corpus import tokenizer_fingerprint
from .text_warmup import _TokenWindow


CACHE_SCHEMA = "natlang.prepared-direct-windows/1"
# Per-window supervision descriptors are part of the cached row identity. Older
# entries can contain only token/role windows, so they cannot safely recover
# masks when identical text has multiple differently annotated rows.
PREPARATION_POLICY = "direct-text-windows/2"
SHARD_BYTES = 32 << 20
TOKEN_BYTES = array("I").itemsize


def _sha_json(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_digest(path):
    if not path:
        return None
    path = Path(path)
    return sha256_file_hex_cached(path) if path.is_file() else None


def cache_spec(engine, args, text_data):
    """Return exact semantic inputs, or None if the tokenizer cannot be fingerprinted."""
    try:
        tokenizer = engine.tokenizer
        from .text_warmup import ROLE_CODES

        def token_id(text):
            try:
                value = tokenizer.convert_tokens_to_ids(text)
                return value if isinstance(value, int) and value != tokenizer.unk_token_id else None
            except Exception:
                return None

        mix = getattr(args, "e2e_mix", None)
        mix_spans = (mix or {}).get("neuralese_spans") if isinstance(mix, dict) else None
        recovery = getattr(args, "recovery_selection", None)
        recovery_path = recovery.get("path") if isinstance(recovery, dict) else recovery
        recovery_binding = recovery.get("sha256") if isinstance(recovery, dict) else None
        # A pinned annotation is part of the rows' meaning. Never allow an
        # absent or stale sidecar to collapse to the same ``None`` digest as a
        # cache entry made without that sidecar. The normal row loader will
        # report the actionable input error after this deliberately uncached
        # path is selected.
        recovery_digest = _file_digest(recovery_path)
        if recovery is not None and (recovery_digest is None or
                                     (recovery_binding is not None and recovery_digest != recovery_binding)):
            return None
        spans_path = (mix_spans or {}).get("path") if isinstance(mix_spans, dict) else mix_spans
        if spans_path and not Path(spans_path).is_absolute():
            from ..common.paths import root as path_root
            spans_path = Path(path_root("repo")) / spans_path
        spans_digest = _file_digest(spans_path)
        spans_binding = (mix_spans or {}).get("sha256") if isinstance(mix_spans, dict) else None
        if mix_spans and (spans_digest is None or
                          (spans_binding is not None and spans_digest != spans_binding)):
            return None
        return {
            "policy": PREPARATION_POLICY,
            "input_sha256": sha256_file_hex_cached(text_data),
            "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
            "vocab_size": len(tokenizer),
            "token_storage": {"bytes_per_id": TOKEN_BYTES, "byteorder": sys.byteorder},
            "control_ids": {
                "open": engine.backbone.controls.open_id,
                "close": engine.backbone.controls.close_id,
                "im_start": token_id("<|im_start|>"),
                "roles": {name: token_id(name) for name in ("system", "user", "assistant", "tool")},
                "think_open": token_id("<think>"),
                "think_close": token_id("</think>"),
                "role_codes": list(ROLE_CODES),
                "tool_response_prefixes": [list(engine._tokens(prefix)) for prefix in
                                            ("\n<tool_response>", "\n\n<tool_response>")],
            },
            "windowing": {
                name: getattr(args, name, default)
                for name, default in (("tokens", None), ("prefix_tokens", None), ("target_tokens", None),
                                      ("train_target_tokens", None), ("train_min_context_tokens", 0),
                                      ("held_documents", None), ("merge_prefix_rows", False))
            },
            "mask_system_prompt": True,
            "row_policy": "stream-text-rows/1",
            "merge_policy": "merge-prefix-rows/1" if getattr(args, "merge_prefix_rows", False) else None,
            "recovery_selection_sha256": recovery_digest,
            "recovery_selection_binding_sha256": recovery_binding,
            "recovery_selection_policy": ({"rules": recovery.get("rules"),
                                            "rendering": recovery.get("rendering")}
                                           if isinstance(recovery, dict) else None),
            "neuralese_spans_sha256": spans_digest,
            "neuralese_spans_binding_sha256": spans_binding,
            "neuralese_routing_policy": "neuralese-output-routing/2" if mix_spans else None,
        }
    except Exception:
        return None


def cache_path(spec):
    if spec is None:
        return None
    root = Path(os.environ.get("NATLANG_PREPARED_WINDOW_CACHE") or row_index.cache_dir() / "prepared-windows")
    return root / _sha_json(spec)


@contextmanager
def locked(path):
    """Serialize preparation and replacement for one content key."""
    if path is None:
        yield False
        return
    lock_path = Path(path).with_name(Path(path).name + ".lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")
    except OSError:
        yield False
        return
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError:
            yield False
            return
        yield True


def _window_record(window, token_refs, role_refs):
    ids = window["ids"]
    token_shard, token_base, _ = token_refs[id(ids._source)]
    record = {key: value for key, value in window.items() if key not in ("ids", "roles")}
    record["_token"] = [token_shard, token_base + ids._start, ids._end - ids._start]
    roles = window.get("roles")
    if roles is not None:
        role_shard, role_base, _ = role_refs[id(roles._source)]
        record["_role"] = [role_shard, role_base + roles._start, roles._end - roles._start]
    return record


def _encode_rows(rows):
    encoded = []
    for row in rows:
        value = dict(row)
        ids = value.get("token_ids")
        if isinstance(ids, range):
            value["token_ids"] = {"_prepared_window_range": [ids.start, ids.stop, ids.step]}
        encoded.append(value)
    return encoded


def _decode_rows(rows):
    decoded = []
    for row in rows:
        value = dict(row)
        ids = value.get("token_ids")
        if isinstance(ids, dict) and set(ids) == {"_prepared_window_range"}:
            value["token_ids"] = range(*ids["_prepared_window_range"])
        decoded.append(value)
    return decoded


def save(path, spec, windows, rows, receipt, held_selection):
    """Commit a complete cache entry atomically. Returns False on ordinary cache I/O failures."""
    if path is None or spec is None:
        return False
    path = Path(path)
    temp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=path.name + ".tmp-", dir=path.parent))
        # Each source row contributes one token source and, when ChatML role
        # labels are active, one role source. Windows point into those sources,
        # so overlapping context is stored once.
        token_refs, role_refs = {}, {}
        files = []

        def write_sources(field, refs, suffix):
            shard = 0
            stream = None
            used = 0
            def close_shard():
                nonlocal stream, used
                if stream is None:
                    return
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                file = temp / f"{suffix}-{shard:05d}.bin"
                files.append({"name": file.name, "bytes": file.stat().st_size,
                              "sha256": sha256_file_hex(file)})
                stream = None
                used = 0
            for split in ("train", "test"):
                for window in windows[split]:
                    obj = window.get(field)
                    if obj is None:
                        continue
                    source = obj._source
                    identity = id(source)
                    if identity in refs:
                        continue
                    raw = memoryview(source).cast("B")
                    if stream is None or (used and used + len(raw) > SHARD_BYTES):
                        close_shard()
                        shard += 1
                        stream = open(temp / f"{suffix}-{shard:05d}.bin", "wb")
                    offset = used
                    stream.write(raw)
                    used += len(raw)
                    width = TOKEN_BYTES if suffix == "tokens" else 1
                    refs[identity] = (shard, offset // width, len(raw) // width)
            close_shard()

        write_sources("ids", token_refs, "tokens")
        write_sources("roles", role_refs, "roles")

        # Preserve shard ordering in separate token/role maps; indices are
        # one-based because shard zero is intentionally never created.
        token_files = sorted((f for f in files if f["name"].startswith("tokens-")), key=lambda f: f["name"])
        role_files = sorted((f for f in files if f["name"].startswith("roles-")), key=lambda f: f["name"])
        token_index = {int(f["name"].split("-")[1].split(".")[0]): i for i, f in enumerate(token_files)}
        role_index = {int(f["name"].split("-")[1].split(".")[0]): i for i, f in enumerate(role_files)}
        token_refs = {key: (token_index[value[0]], value[1], value[2]) for key, value in token_refs.items()}
        role_refs = {key: (role_index[value[0]], value[1], value[2]) for key, value in role_refs.items()}
        if os.environ.get("NATLANG_PREPARED_WINDOW_CACHE_DEBUG"):
            print(f"prepared-window shards: {token_index=} {list(token_refs.values())[:2]}", flush=True)
        records = {split: [_window_record(window, token_refs, role_refs) for window in windows[split]]
                   for split in ("train", "test")}
        metadata_path = temp / "windows.json"
        metadata = {"records": records, "rows": _encode_rows(rows), "receipt": receipt,
                    "held_selection": held_selection}
        with open(metadata_path, "wb") as handle:
            handle.write(json.dumps(metadata, ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        files.append({"name": metadata_path.name, "bytes": metadata_path.stat().st_size,
                      "sha256": sha256_file_hex(metadata_path)})
        manifest = {"schema": CACHE_SCHEMA, "spec_sha256": _sha_json(spec), "spec": spec,
                    "files": files, "windows": {split: len(windows[split]) for split in ("train", "test")},
                    "target_windows": sum(1 for window in windows["train"] if _target_bearing(window))}
        manifest_path = temp / "manifest.json"
        with open(manifest_path, "wb") as handle:
            handle.write(canonical_json_bytes(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            # Another process may have completed the same content key while
            # this process was preparing. Never replace a published entry.
            shutil.rmtree(temp, ignore_errors=True)
            return False
        os.replace(temp, path)
        return True
    except Exception:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
        return False


def _target_bearing(window):
    from .text_supervision import target_bearing_window
    return target_bearing_window(window)


def load(path, spec):
    """Validate an entry and restore windows on read-only maps; None means cache miss."""
    if path is None:
        return None
    path = Path(path)
    maps = []
    def miss(reason):
        if os.environ.get("NATLANG_PREPARED_WINDOW_CACHE_DEBUG"):
            print(f"prepared-window cache miss: {reason}", flush=True)
        return None
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if (manifest.get("schema") != CACHE_SCHEMA or manifest.get("spec") != spec or
                manifest.get("spec_sha256") != _sha_json(spec)):
            return miss("schema or semantic key mismatch")
        file_records = {record["name"]: record for record in manifest["files"]}
        for name, record in file_records.items():
            if not isinstance(name, str) or Path(name).name != name:
                return miss("invalid manifest filename")
            file = path / name
            if not file.is_file() or file.stat().st_size != record["bytes"] or sha256_file_hex(file) != record["sha256"]:
                return miss("missing file, size mismatch, or digest mismatch: " + str(name))
        metadata = json.loads((path / "windows.json").read_text())
        token_files = sorted((name for name in file_records if name.startswith("tokens-")))
        role_files = sorted((name for name in file_records if name.startswith("roles-")))
        token_maps = []
        for name in token_files:
            handle = open(path / name, "rb")
            mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            handle.close()
            maps.append(mm)
            token_maps.append(memoryview(mm).cast("I"))
        role_maps = []
        for name in role_files:
            handle = open(path / name, "rb")
            mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            handle.close()
            maps.append(mm)
            role_maps.append(memoryview(mm))
        restored = {}
        for split in ("train", "test"):
            values = []
            for record in metadata["records"][split]:
                record = dict(record)
                shard, start, length = record.pop("_token")
                if (type(shard) is not int or type(start) is not int or type(length) is not int or
                        shard < 0 or shard >= len(token_maps) or start < 0 or length < 0 or
                        start + length > len(token_maps[shard])):
                    return miss(f"token window descriptor outside backing shard: {shard=} {start=} {length=} "
                                f"shards={len(token_maps)} shard_len="
                                f"{len(token_maps[shard]) if type(shard) is int and 0 <= shard < len(token_maps) else None}")
                record["ids"] = _TokenWindow(token_maps[shard], start, start + length)
                role = record.pop("_role", None)
                if role is not None:
                    shard, start, length = role
                    if (type(shard) is not int or type(start) is not int or type(length) is not int or
                            shard < 0 or shard >= len(role_maps) or start < 0 or length < 0 or
                            start + length > len(role_maps[shard])):
                        return miss("role window descriptor outside backing shard")
                    record["roles"] = _TokenWindow(role_maps[shard], start, start + length)
                for key in ("supervised_spans",):
                    if key in record:
                        record[key] = [tuple(item) for item in record[key]]
                values.append(record)
            restored[split] = values
        if any(len(restored[split]) != manifest["windows"][split] for split in restored):
            return miss("window count mismatch")
        if sum(1 for window in restored["train"] if _target_bearing(window)) != manifest["target_windows"]:
            return miss("target-bearing count mismatch")
        # Retain the maps through their views; the memoryviews keep each mmap
        # alive after this local list leaves scope.
        if (metadata["receipt"].get("documents") != len(metadata["rows"]) or
                metadata["receipt"].get("masked_system_tokens", -1) < 0):
            return miss("row/mask receipt mismatch")
        return restored, _decode_rows(metadata["rows"]), metadata["receipt"], metadata["held_selection"]
    except Exception as error:
        return miss(type(error).__name__ + ": " + str(error))
