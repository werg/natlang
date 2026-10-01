#!/usr/bin/env python3
"""Create a lineage-preserving, lane-qualified view of reviewed legacy turns.

Source files and their root approval manifests stay immutable. The derived view
only changes the row ID and adds reviewed_legacy_identity metadata so two
approved runtime-API lanes that reused a turn ID remain distinct downstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


VERSION = "natlang.reviewed_legacy_turn_identity/1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, str, str]:
    # Check symlink policy on the arguments before resolve(), which follows links.
    source_arg = Path(args.input)
    manifest_arg = Path(args.review_manifest)
    output_arg = Path(args.output)
    manifest_out_arg = Path(args.manifest_out)
    for path in (source_arg, manifest_arg, output_arg, manifest_out_arg):
        if path.is_symlink():
            raise ValueError(f"symlink path is not allowed: {path}")
    root = args.root.resolve()
    source = source_arg.resolve()
    manifest_path = manifest_arg.resolve()
    output = output_arg.resolve()
    manifest_out = manifest_out_arg.resolve()
    if not source.is_relative_to(root) or not manifest_path.is_relative_to(root):
        raise ValueError("source and review manifest must remain inside repository root")
    if len({source, manifest_path, output, manifest_out}) != 4:
        raise ValueError("source, review manifest, output and output manifest must be distinct paths")
    if not source.is_file() or not manifest_path.is_file():
        raise ValueError("reviewed source or review manifest is missing")
    if output.exists() and output.is_dir():
        raise ValueError("derived output path is a directory")
    if manifest_out.exists() and manifest_out.is_dir():
        raise ValueError("output manifest path is a directory")
    source_relative = source.relative_to(root).as_posix()
    manifest_relative = manifest_path.relative_to(root).as_posix()
    if source_relative != args.source_relative:
        raise ValueError("resolved source path differs from pinned relative source path")
    if source.parent.name != args.lane:
        raise ValueError("lane identifier must match the approved replacement directory")
    if not args.lane or "/" in args.lane or "\\" in args.lane:
        raise ValueError("lane must be a nonempty path-safe identifier")
    return root, source, manifest_path, output, manifest_out, source_relative, manifest_relative


def derive(args: argparse.Namespace, paths: tuple[Path, Path, Path, Path, Path, str, str]) -> dict:
    root, source, manifest_path, output, manifest_out, source_relative, manifest_relative = paths
    source_sha = digest_file(source)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = sha256(manifest_bytes)
    if source_sha != args.expected_file_sha256:
        raise ValueError("reviewed source file SHA-256 changed")
    if manifest_sha != args.expected_manifest_sha256:
        raise ValueError("review manifest SHA-256 changed")
    manifest = json.loads(manifest_bytes)
    if (manifest.get("status") != "source_reviewed_replacement_approved" or
            manifest.get("replacement_artifact") != source_relative or
            manifest.get("replacement_sha256") != source_sha):
        raise ValueError("review manifest does not approve this exact replacement file")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".pending", dir=output.parent)
    temp_path = Path(temp_name)
    row_count = 0
    seen_ids: set[str] = set()
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    try:
        with source.open("rb") as input_stream, os.fdopen(fd, "wb") as output_stream:
            for line_number, raw_line in enumerate(input_stream, start=1):
                input_digest.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except Exception as exc:
                    raise ValueError(f"malformed JSON row at line {line_number}") from exc
                if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                    raise ValueError(f"row {line_number} must be an object with a nonempty string id")
                if "reviewed_legacy_identity" in row:
                    raise ValueError(f"row {line_number} already has reviewed_legacy_identity; refusing reapplication")
                original_id = row["id"]
                row_sha = sha256(raw_line)
                identity = {
                    "version": VERSION,
                    "lane": args.lane,
                    "source_path": source_relative,
                    "source_file_sha256": source_sha,
                    "review_manifest_path": manifest_relative,
                    "review_manifest_sha256": manifest_sha,
                    "original_id": original_id,
                    "source_row_sha256": row_sha,
                }
                key_sha = sha256(canonical(identity))
                derived_id = f"reviewed-legacy:{args.lane}:{key_sha[:24]}"
                if derived_id in seen_ids:
                    raise ValueError(f"derived identity collision at source line {line_number}")
                seen_ids.add(derived_id)
                row["id"] = derived_id
                row["reviewed_legacy_identity"] = {**identity, "identity_sha256": key_sha}
                encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
                output_stream.write(encoded)
                output_digest.update(encoded)
                row_count += 1
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if input_digest.hexdigest() != source_sha:
            raise ValueError("source changed while streaming reviewed rows")
        os.replace(temp_path, output)
        dir_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return {
        "version": VERSION,
        "lane": args.lane,
        "source_path": source_relative,
        "source_file_sha256": source_sha,
        "review_manifest_path": manifest_relative,
        "review_manifest_sha256": manifest_sha,
        "output_path": output.relative_to(root).as_posix() if output.is_relative_to(root) else str(output),
        "output_manifest_path": manifest_out.relative_to(root).as_posix() if manifest_out.is_relative_to(root) else str(manifest_out),
        "output_sha256": output_digest.hexdigest(),
        "rows": row_count,
        "identity_key_fields": ["lane", "review_manifest_sha256", "source_file_sha256", "original_id", "source_row_sha256"],
        "source_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-relative", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()
    paths = validate_paths(args)
    result = derive(args, paths)
    manifest_out = paths[4]
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{manifest_out.name}.", suffix=".pending", dir=manifest_out.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_out)
        dir_fd = os.open(manifest_out.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
