#!/usr/bin/env python3
"""Acquire immutable Hugging Face dataset Parquet files into resumable raw JSONL shards.

This reads published Parquet files directly and never runs dataset loading scripts.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import signal
import sys
import uuid
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable

SCHEMA = "natlang.hf_parquet_acquisition/1"
DEFAULT_BATCH_ROWS = 1000


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with staged.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    staged.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def read_token_file(path: str | Path) -> str:
    token_path = Path(path).expanduser().resolve()
    info = token_path.stat()
    if not token_path.is_file() or (info.st_mode & 0o077) != 0 or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        raise ValueError("--token-file must be a regular file owned by this user with permissions no broader than 0600")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError("--token-file must contain one nonempty token")
    return token


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # PyArrow scalar wrappers expose as_py(); keep the underlying value stable.
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        return _json_safe(as_py())
    raise TypeError(f"unsupported Parquet value type: {type(value).__name__}")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(json.dumps(_json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in rows)).encode("utf-8")


def matching_parquet_files(files: list[str], *, config: str, split: str) -> list[str]:
    """Select files in a convert/parquet layout without accidentally mixing configs/splits."""
    selected = []
    for name in files:
        path = PurePosixPath(name)
        if path.suffix.lower() != ".parquet":
            continue
        parts = [part.lower() for part in path.parts]
        # Convert branches conventionally use <config>/<split>-NNNN.parquet,
        # but also accept <config>/<split>/part-*.parquet.
        split_dir = next((index for index, part in enumerate(parts[:-1]) if part == split.lower()), None)
        split_file = bool(re.match(rf"^{re.escape(split.lower())}(?:[-_.]|$)", path.name.lower()))
        if split_dir is None and not split_file:
            continue
        config_prefix = parts[:split_dir] if split_dir is not None else parts[:-1]
        config_prefix = [part for part in config_prefix if part != "data"]
        if config.lower() == "default":
            if config_prefix not in ([], ["default"]):
                continue
        elif config.lower() not in config_prefix:
            continue
        selected.append(name)
    return sorted(selected)


def _file_size_map(info: Any) -> dict[str, int | None]:
    result = {}
    for sibling in getattr(info, "siblings", []) or []:
        name = getattr(sibling, "rfilename", None)
        if name:
            size = getattr(sibling, "size", None)
            result[name] = int(size) if isinstance(size, (int, float)) else None
    return result


def _load_runtime(token: str | None):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as parquet
    return HfApi(token=token), hf_hub_download, parquet


class _StopRequested(Exception):
    pass


def acquire_parquet(*, out: str | Path, source: str, dataset: str, config: str,
                    split: str = "train", limit: int = 25000, revision: str = "refs/convert/parquet",
                    batch_rows: int = DEFAULT_BATCH_ROWS, token: str | None = None,
                    api: Any = None, downloader: Callable[..., str] | None = None,
                    parquet_module: Any = None, should_stop: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    if not all((source, dataset, config, split, revision)):
        raise ValueError("source, dataset, config, split, and revision must be nonempty")
    if not isinstance(limit, int) or limit < 1 or not isinstance(batch_rows, int) or batch_rows < 1:
        raise ValueError("limit and batch_rows must be positive integers")
    if api is None or downloader is None or parquet_module is None:
        default_api, default_downloader, default_parquet = _load_runtime(token)
        api = api or default_api
        downloader = downloader or default_downloader
        parquet_module = parquet_module or default_parquet

    output = Path(out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    base_identity = {"source": source, "dataset": dataset, "config": config, "split": split,
                     "requested_revision": revision, "batch_rows": batch_rows}
    manifest = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    if manifest is not None:
        if manifest.get("identity") != base_identity:
            raise ValueError("existing acquisition config differs from manifest")
        if limit < manifest.get("limit", limit):
            raise ValueError("limit cannot decrease on resume")
        for shard in manifest.get("shards", []):
            raw_path = output / shard.get("path", shard.get("raw"))
            if not raw_path.is_file() or _file_sha(raw_path) != shard.get("sha256"):
                raise ValueError(f"committed raw shard missing or hash mismatch: {raw_path.name}")

    info = api.dataset_info(dataset, revision=revision, files_metadata=True)
    resolved_sha = getattr(info, "sha", None)
    if not isinstance(resolved_sha, str) or not resolved_sha:
        raise ValueError("Hugging Face dataset metadata returned no resolved commit SHA")
    files = api.list_repo_files(repo_id=dataset, repo_type="dataset", revision=resolved_sha)
    sizes = _file_size_map(info)
    selected = matching_parquet_files(files, config=config, split=split)
    if not selected:
        raise ValueError(f"no Parquet files found for config={config!r}, split={split!r} at revision {resolved_sha}")
    selected_info = [{"path": name, "size_bytes": sizes.get(name)} for name in selected]

    if manifest is not None and manifest.get("resolved_revision") != resolved_sha:
        raise ValueError("dataset revision changed since the acquisition manifest was created")
    if manifest is None:
        manifest = {"schema": SCHEMA, "identity": base_identity, "source": source, "dataset": dataset,
                    "requested_revision": revision, "revision": resolved_sha, "resolved_revision": resolved_sha,
                    "files": selected_info, "limit": limit, "next_file_index": 0,
                    "next_row_in_file": 0, "rows_committed": 0, "shards": [], "status": "pending"}
    elif manifest.get("files") != selected_info:
        raise ValueError("Parquet file list or sizes changed at the pinned dataset revision")
    manifest.pop("error", None)
    manifest["limit"] = limit
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def commit():
        _atomic_json(manifest_path, manifest)

    commit()
    if manifest["rows_committed"] >= limit:
        manifest["status"] = "complete"
        commit()
        return manifest
    try:
        for file_index in range(manifest["next_file_index"], len(selected)):
            if should_stop():
                raise _StopRequested()
            parquet_name = selected[file_index]
            local = Path(downloader(repo_id=dataset, filename=parquet_name, repo_type="dataset",
                                    revision=resolved_sha, token=token)).resolve()
            parquet_sha = _file_sha(local)
            prior_hashes = {item["source_file_sha256"] for item in manifest["shards"]
                            if item["source_file"] == parquet_name}
            if prior_hashes and prior_hashes != {parquet_sha}:
                raise ValueError(f"downloaded Parquet hash changed for {parquet_name}")
            parquet_file = parquet_module.ParquetFile(local)
            metadata_rows = int(parquet_file.metadata.num_rows)
            if manifest["next_row_in_file"] > metadata_rows:
                raise ValueError(f"resume row offset exceeds Parquet file row count: {parquet_name}")
            offset = 0
            for batch in parquet_file.iter_batches(batch_size=batch_rows):
                batch_dicts = batch.to_pylist()
                batch_start, batch_end = offset, offset + len(batch_dicts)
                offset = batch_end
                if batch_end <= manifest["next_row_in_file"]:
                    continue
                if batch_start > manifest["next_row_in_file"]:
                    raise ValueError(f"Parquet batch boundaries changed while resuming {parquet_name}")
                if should_stop():
                    raise _StopRequested()
                consumed_start = max(batch_start, manifest["next_row_in_file"])
                batch_dicts = batch_dicts[consumed_start - batch_start:]
                remaining = limit - manifest["rows_committed"]
                if remaining <= 0:
                    manifest["status"] = "complete"
                    commit()
                    return manifest
                batch_dicts = batch_dicts[:remaining]
                raw = _jsonl_bytes(batch_dicts)
                shard_index = len(manifest["shards"])
                raw_name = f"part-{shard_index:08d}.jsonl"
                _atomic_write(output / raw_name, raw)
                consumed_end = consumed_start + len(batch_dicts)
                manifest["shards"].append({"index": shard_index, "path": raw_name, "source_file": parquet_name,
                    "source_file_sha256": parquet_sha, "source_row_start": consumed_start,
                    "source_row_end": consumed_end, "rows": len(batch_dicts),
                    "sha256": _sha(raw)})
                manifest["rows_committed"] += len(batch_dicts)
                manifest["next_row_in_file"] = consumed_end
                if consumed_end >= metadata_rows:
                    manifest["next_file_index"] = file_index + 1
                    manifest["next_row_in_file"] = 0
                else:
                    manifest["next_file_index"] = file_index
                manifest["status"] = "paused" if should_stop() else "running"
                commit()
                if should_stop():
                    manifest["status"] = "paused"
                    commit()
                    return manifest
                if manifest["rows_committed"] >= limit:
                    manifest["status"] = "complete"
                    commit()
                    return manifest
            if offset != metadata_rows:
                raise ValueError(f"Parquet iterator row count does not match metadata: {parquet_name}")
            manifest["next_file_index"] = file_index + 1
            manifest["next_row_in_file"] = 0
            commit()
        manifest["status"] = "complete"
        commit()
        return manifest
    except _StopRequested:
        manifest["status"] = "paused"
        commit()
        return manifest
    except Exception as exc:
        manifest["status"] = "error"
        message = str(exc)
        if token:
            message = message.replace(token, "[redacted]")
        manifest["error"] = message
        commit()
        return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=25000)
    parser.add_argument("--revision", default="refs/convert/parquet")
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args(argv)
    token = read_token_file(args.token_file) if args.token_file else None
    stopping = False
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    old_int, old_term = signal.signal(signal.SIGINT, stop), signal.signal(signal.SIGTERM, stop)
    try:
        manifest = acquire_parquet(out=args.out, source=args.source, dataset=args.dataset, config=args.config,
            split=args.split, limit=args.limit, revision=args.revision, batch_rows=args.batch_rows,
            token=token, should_stop=lambda: stopping)
        print(json.dumps({"source": args.source, "rows": manifest.get("rows_committed", 0),
                          "status": manifest.get("status"), "files": len(manifest.get("files", []))}))
        return 75 if stopping or manifest.get("status") == "paused" else 1 if manifest.get("status") == "error" else 0
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
