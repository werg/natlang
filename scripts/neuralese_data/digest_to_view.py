#!/usr/bin/env python3
"""The registered conversion of records written before the view rename (plans/neuralese/DECISIONS.md 2026-10-09, "one
summarizer family": `view` replaces `digest` everywhere, by rewrite).

Records from natlang.neuralese-conversion/14 and earlier (ts-host neuralese-conversion.ts) and
natlang.harness-bench-conversion/1 (harness_bench/records.py) name the listing view a `digest` part; the trainer
refuses them. This conversion writes a new corpus directory (published corpora are immutable) in which:

- every message part `{"type": "digest", "name": "digest:<sha12>", ...}` is `{"type": "view", "name": "view:<sha12>",
  ...}`, its other fields unchanged, and a `note` that reads "  // digest of the ..." reads "  // view of the ...";
- `neuralese_conversion.sites.digest` is `sites.view`, the harness-bench version is
  natlang.harness-bench-conversion/2 (`digests_in_trajectory` → `views_in_trajectory`, provenance `digest_chars` →
  `view_chars`), and each record carries `view_rename` naming this conversion and the version it came from;
- the piece `prompt:digest` (the digest operator's instructions) is `prompt:view` with view's body
  (natlang_neuralese.view.INSTRUCTIONS), as a soft parameter initialised from its new text;
- every other file named with `--copy` is copied byte for byte.

`conversion.json` records the input and output SHA-256 and the counts. Register the output under a new corpus id with
its own manifest (scripts/sync_training_corpora.py); admission is decided there, never by this script.

Usage: digest_to_view.py --source DIR --out DIR [--copy summary.json surface.json ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
from natlang_neuralese.view import INSTRUCTIONS  # noqa: E402

CONVERTER = "scripts/neuralese_data/digest_to_view.py@1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_part(part: dict) -> tuple[dict, bool]:
    if part.get("type") != "digest":
        return part, False
    out = {**part, "type": "view"}
    if isinstance(out.get("name"), str) and out["name"].startswith("digest:"):
        out["name"] = "view:" + out["name"][len("digest:"):]
    if isinstance(out.get("note"), str) and out["note"].startswith("  // digest of the "):
        out["note"] = "  // view of the " + out["note"][len("  // digest of the "):]
    return out, True


def convert_record(record: dict) -> tuple[dict, int]:
    parts = 0
    messages = []
    for message in record.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            converted = [convert_part(part) for part in content]
            parts += sum(changed for _, changed in converted)
            message = {**message, "content": [part for part, _ in converted]}
        messages.append(message)
    out = {**record, "messages": messages}
    conversion = dict(out.get("neuralese_conversion") or {})
    previous = conversion.get("version")
    sites = dict(conversion.get("sites") or {})
    if "digest" in sites:
        sites["view"] = sites.pop("digest")
        conversion["sites"] = sites
    if previous == "natlang.harness-bench-conversion/1":
        conversion["version"] = "natlang.harness-bench-conversion/2"
    if conversion:
        out["neuralese_conversion"] = conversion
    if "digests_in_trajectory" in out:
        out["views_in_trajectory"] = out.pop("digests_in_trajectory")
    provenance = out.get("provenance")
    if isinstance(provenance, dict) and "digest_chars" in provenance:
        provenance = dict(provenance)
        provenance["view_chars"] = provenance.pop("digest_chars")
        out["provenance"] = provenance
    out["view_rename"] = {"converter": CONVERTER, "from_version": previous, "digest_parts": parts}
    return out, parts


def convert_piece(piece: dict) -> dict:
    if piece.get("name") == "prompt:digest":
        return {**piece, "name": "prompt:view", "text": INSTRUCTIONS}
    return piece


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, help="corpus directory with records.jsonl and pieces.jsonl")
    parser.add_argument("--out", required=True, help="new directory (must not exist)")
    parser.add_argument("--records", default="records.jsonl")
    parser.add_argument("--pieces", default="pieces.jsonl")
    parser.add_argument("--copy", nargs="*", default=[], help="other files copied unchanged")
    args = parser.parse_args(argv)
    source, out = Path(args.source), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a conversion writes a new directory")
    out.mkdir(parents=True)
    counts = {"records": 0, "records_with_digest_parts": 0, "digest_parts": 0, "pieces_renamed": 0}
    with (source / args.records).open() as lines, (out / args.records).open("w") as handle:
        for line in lines:
            if not line.strip():
                continue
            record, parts = convert_record(json.loads(line))
            counts["records"] += 1
            counts["digest_parts"] += parts
            counts["records_with_digest_parts"] += bool(parts)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if (source / args.pieces).exists():
        with (source / args.pieces).open() as lines, (out / args.pieces).open("w") as handle:
            for line in lines:
                if line.strip():
                    piece = json.loads(line)
                    converted = convert_piece(piece)
                    counts["pieces_renamed"] += converted is not piece
                    handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
    for name in args.copy:
        shutil.copy2(source / name, out / name)
    files = [args.records] + ([args.pieces] if (source / args.pieces).exists() else []) + list(args.copy)
    receipt = {"converter": CONVERTER, "decision": "plans/neuralese/DECISIONS.md 2026-10-09 one summarizer family",
               "source": str(source), "counts": counts,
               "input_sha256": {name: sha256(source / name) for name in files},
               "output_sha256": {name: sha256(out / name) for name in files}}
    (out / "conversion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
