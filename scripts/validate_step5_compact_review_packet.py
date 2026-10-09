#!/usr/bin/env python3
"""Validate immutable inputs, commands, renderer snapshot, and outputs in a compact packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pin(pin: dict[str, Any]) -> Path:
    rel = pin.get("path")
    if not isinstance(rel, str) or not rel:
        raise ValueError("artifact pin has no path")
    path = (ROOT / rel).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"pinned artifact is missing or outside repository: {rel}")
    actual_bytes = path.stat().st_size
    actual_hash = sha256(path)
    if pin.get("bytes") != actual_bytes or pin.get("sha256") != actual_hash:
        raise ValueError(f"pinned artifact bytes/hash mismatch: {rel}")
    return path


def option(argv: list[str], flag: str, *, required: bool = True) -> str | None:
    indexes = [i for i, value in enumerate(argv) if value == flag]
    if not indexes and not required:
        return None
    if len(indexes) != 1 or indexes[0] + 1 >= len(argv):
        raise ValueError(f"command must contain exactly one value for {flag}")
    return argv[indexes[0] + 1]


def verify_compact_packet(packet: dict[str, Any]) -> dict[str, int]:
    if packet.get("schema") != "natlang.step5-compact-native-text-review-packet/1":
        raise ValueError("unsupported compact review packet schema")
    all_pins: dict[str, dict[str, Any]] = {}
    for group in (packet.get("artifact_pins", []),
                  packet.get("native_delta", {}).get("inputs", []),
                  packet.get("native_delta", {}).get("outputs", []),
                  packet.get("text_delta", {}).get("inputs", []),
                  packet.get("text_delta", {}).get("outputs", [])):
        for pin in group:
            path = pin.get("path")
            core = {key: pin.get(key) for key in ("path", "bytes", "sha256")}
            if path in all_pins and {key: all_pins[path].get(key) for key in core} != core:
                raise ValueError(f"conflicting pins for {path}")
            all_pins[path] = core
    for pin in all_pins.values():
        verify_pin(pin)

    renderer = packet.get("renderer", {})
    snapshot_pin = renderer.get("renderer_snapshot_manifest")
    if not isinstance(snapshot_pin, dict):
        raise ValueError("packet has no renderer snapshot manifest pin")
    snapshot_path = verify_pin(snapshot_pin)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    file_map = snapshot.get("files")
    if not isinstance(file_map, dict):
        raise ValueError("renderer snapshot manifest has no file hash map")

    argv = packet.get("commands", {}).get("text_builder_container_argv")
    if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv):
        raise ValueError("packet lacks the exact text-builder container argv")
    mounts = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--mount"]
    repo_mounts = [mount for mount in mounts if "dst=/repo" in mount]
    output_mounts = [mount for mount in mounts if "dst=/out" in mount]
    if len(repo_mounts) != 1 or "src=" + str(ROOT) not in repo_mounts[0] or "readonly" not in repo_mounts[0]:
        raise ValueError("text-builder command must bind the canonical repo read-only at /repo")
    if len(output_mounts) != 1:
        raise ValueError("text-builder command must have one /out bind mount")
    builder = option(argv, "python")
    if not builder or not builder.startswith("/repo/"):
        raise ValueError("text-builder command does not name a repo-mounted snapshot builder")
    builder_rel = builder.removeprefix("/repo/")
    snapshot_rel = str(Path(snapshot_pin["path"]).parent)
    if not builder_rel.startswith(snapshot_rel + "/"):
        raise ValueError("builder path is outside the packet's pinned renderer snapshot")
    inner_builder = builder_rel[len(snapshot_rel) + 1:]
    if inner_builder not in file_map:
        raise ValueError("builder is not listed in the renderer snapshot manifest")
    builder_pin = verify_pin({"path": f"{snapshot_rel}/{inner_builder}",
                              "bytes": (ROOT / snapshot_rel / inner_builder).stat().st_size,
                              "sha256": file_map[inner_builder]})
    if builder_pin != (ROOT / snapshot_rel / inner_builder).resolve():
        raise ValueError("builder snapshot resolution mismatch")
    renderer_root = option(argv, "--renderer-package-root")
    if renderer_root != f"/repo/{snapshot_rel}/training/neuralese":
        raise ValueError("renderer package root does not match the pinned snapshot")
    if argv.count("--compact-only") != 1:
        raise ValueError("text-builder command lacks --compact-only")
    if option(argv, "--repo-root") != "/repo":
        raise ValueError("text-builder command repo root is not /repo")
    if option(argv, "--out") != "/out":
        raise ValueError("text-builder command output is not /out")
    expected_tokenizer = "/repo/" + renderer.get("tokenizer_path", "")
    if option(argv, "--tokenizer") != expected_tokenizer:
        raise ValueError("text-builder tokenizer path does not match renderer metadata")
    out_mount = output_mounts[0]
    fields = dict(part.split("=", 1) for part in out_mount.split(",") if "=" in part)
    output_src = fields.get("src")
    text_outputs = packet.get("text_delta", {}).get("outputs", [])
    if not output_src or not text_outputs:
        raise ValueError("text delta has no output mount or output pins")
    expected_out = (ROOT / text_outputs[0]["path"]).resolve().parent
    if Path(output_src).resolve() != expected_out:
        raise ValueError("text-builder output mount does not match the pinned delta output directory")
    ancillary = {snapshot_path.resolve()}
    if any((ROOT / item["path"]).resolve().parent != expected_out and
           (ROOT / item["path"]).resolve() not in ancillary for item in text_outputs):
        raise ValueError("text output pins span directories not represented by the single output mount")

    text_inputs = packet["text_delta"].get("inputs", [])
    if not text_inputs or option(argv, "--delta-records") != "/repo/" + text_inputs[0]["path"]:
        raise ValueError("text-builder delta input differs from the first pinned text input")
    pieces_arg = option(argv, "--pieces")
    if pieces_arg is None or not pieces_arg.startswith("/repo/"):
        raise ValueError("text-builder pieces path is not repo-mounted")
    pieces_rel = pieces_arg.removeprefix("/repo/")
    if pieces_rel in all_pins:
        pass
    else:
        pieces_path = (ROOT / pieces_rel).resolve()
        if not pieces_path.is_relative_to(ROOT) or not pieces_path.is_file() or pieces_path.read_bytes() != b"":
            raise ValueError("unpinned text-builder pieces input is not the canonical empty file")
    approval = packet.get("source", {}).get("selected_action_admission")
    if approval and option(argv, "--source-approval") != "/repo/" + approval["path"]:
        raise ValueError("text-builder approval path differs from packet source admission pin")

    native_argv = packet.get("commands", {}).get("native_assembler_argv")
    if not isinstance(native_argv, list):
        raise ValueError("packet lacks exact native assembler argv")
    native_inputs = packet.get("native_delta", {}).get("inputs", [])
    native_paths = {pin["path"] for pin in native_inputs}
    native_delta_path = option(native_argv, "--delta-native")
    if native_delta_path not in native_paths:
        raise ValueError("native assembler delta input is not among pinned native inputs")
    native_outputs = packet.get("native_delta", {}).get("outputs", [])
    if not native_outputs or any((ROOT / item["path"]).resolve().parent !=
                                 (ROOT / native_outputs[0]["path"]).resolve().parent
                                 for item in native_outputs):
        raise ValueError("native output pins do not share one compact output directory")
    if Path(option(native_argv, "--out") or "").resolve() != \
            (ROOT / native_outputs[0]["path"]).resolve().parent:
        raise ValueError("native assembler output directory differs from pinned outputs")
    if option(native_argv, "--approval") not in {approval["path"], "/repo/" + approval["path"]}:
        raise ValueError("native assembler approval path differs from packet source admission pin")
    return {"pins_verified": len(all_pins), "renderer_files": len(file_map),
            "text_outputs": len(text_outputs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True, type=Path)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    summary = verify_compact_packet(packet)
    print(json.dumps({"ok": True, **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
