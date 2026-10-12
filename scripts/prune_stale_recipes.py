#!/usr/bin/env python3
"""Delete discarded Neuralese training recipes (owner decision 2026-10-12, plans/REVIEW_20261012.md rec. 9,
"one recipe lineage"; AGENTS.md "Storage hygiene and discarded work": delete, do not archive).

KEEP is the live lineage and its required parents:
  * every ``neuralese-direct-*.json`` and ``neuralese-e2e-*.json`` (shared bases, Mellum and LFM lines, Pop's Muon
    main run);
  * the E2E arm named by training/neuralese/line-parity.json
    (``luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3``);
  * the transitive closure of what a kept recipe or line-parity.json references by recipe id or file stem through
    ``extends`` or as a parent/checkpoint/artifact source in any structural (non-prose) string value
    (prose such as ``description``/``reason``, a run's ``outputs``, ``limitations`` and ``lineage`` evidence are
    ignored: a sentence or an evidence report naming an old run is not a dependency);
  * HISTORY.md.
Every other file in training/neuralese/recipes/ is listed with its family, size and reason. Nothing is deleted
without ``--execute``.

    python3 scripts/prune_stale_recipes.py              # list the exact delete set
    python3 scripts/prune_stale_recipes.py --execute    # delete it
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "training" / "neuralese" / "recipes"
LINE_PARITY = ROOT / "training" / "neuralese" / "line-parity.json"

SEED_PATTERNS = (re.compile(r"^neuralese-direct-.*\.json$"), re.compile(r"^neuralese-e2e-.*\.json$"))
SEED_FILES = ("luna-direct-lfm-native-e2e-token-writer-endpoint-epoch-20261011-v3.json", "HISTORY.md")
PROSE_KEYS = {"description", "reason", "note", "notes", "comment", "pending", "difference", "history"}
# Not dependencies either: a run's own output directory, its prose limitations, and lineage evidence (reports of
# earlier failed attempts cited as evidence; the artifacts stay where they are, their recipes are not parents).
NON_PARENT_KEYS = PROSE_KEYS | {"limitations", "lineage", "outputs"}

# (family prefix, why it is discarded); the first matching prefix names the family.
FAMILIES = (
    ("foundation-", "token-preserving foundation design (discarded by the direct redesign, owner 2026-10-11)"),
    ("gold-text-", "gold-text warm-up / input-map design (discarded by the direct redesign)"),
    ("latent-sketch-consumer-", "sketch consumer design (replaced 2026-10-08, then discarded)"),
    ("local-stage-sketch-credit-", "sketch credit design (discarded)"),
    ("sketch-training-defaults-", "sketch defaults (discarded)"),
    ("learned-residual-curriculum-", "learned residual curriculum (discarded)"),
    ("semantic-exposure-data-contract-", "semantic exposure contract of the discarded warm-up"),
    ("native-text-recurrence-", "native-text recurrence design (discarded)"),
    ("luna-ar-", "mapped/AR graph design (discarded)"),
    ("raw-recurrence-", "raw-recurrence foundation/warm-up chain (superseded by neuralese-direct; the E2E stage's "
                        "config is inline in line-parity.json)"),
    ("fp8-lfm-direct-", "FP8 comparison diagnostic (concluded; plans/neuralese/NUMERICAL_PARITY.md)"),
    ("luna-direct-lfm-native-e2e-", "superseded native E2E diagnostic/launch"),
    ("luna-neuralese-direct-lfm-base-", "superseded LFM direct base run"),
    ("optimizer-bakeoff-", "optimizer bake-off (concluded; plans/neuralese/OPTIMIZER_BAKEOFF_20261011.md)"),
)


def structural_strings(value, key: str | None = None):
    if key in NON_PARENT_KEYS:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from structural_strings(v, k)
    elif isinstance(value, list):
        for v in value:
            yield from structural_strings(v, key)


def names_of(path: Path) -> set[str]:
    names = {path.stem}
    try:
        recipe_id = json.loads(path.read_text()).get("id")
    except (ValueError, AttributeError):
        recipe_id = None
    if isinstance(recipe_id, str):
        names.add(recipe_id)
    return names


def mentions(text: str, name: str) -> bool:
    return re.search(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])", text) is not None


def keep_closure() -> tuple[set[Path], dict[Path, str]]:
    files = sorted(p for p in RECIPES.iterdir() if p.is_file())
    keep: dict[Path, str] = {}
    for p in files:
        if p.name in SEED_FILES or any(rx.match(p.name) for rx in SEED_PATTERNS):
            keep[p] = "live lineage seed"
    names = {p: names_of(p) for p in files if p.suffix == ".json"}
    queue = [p for p in keep if p.suffix == ".json"] + [LINE_PARITY]
    while queue:
        src = queue.pop()
        data = json.loads(src.read_text())
        text = "\n".join(structural_strings(data))
        for p, ns in names.items():
            if p in keep:
                continue
            hit = next((n for n in sorted(ns) if mentions(text, n)), None)
            if hit:
                keep[p] = f"referenced by {src.name} ({hit})"
                queue.append(p)
    return set(files), keep


def family(name: str) -> tuple[str, str]:
    for prefix, why in FAMILIES:
        if name.startswith(prefix):
            return prefix.rstrip("-"), why
    return "unclassified", "not in the live lineage closure"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="delete the listed files")
    args = ap.parse_args()
    files, keep = keep_closure()
    delete = sorted(files - set(keep))
    print(f"KEEP ({len(keep)}):")
    for p in sorted(keep):
        print(f"  {p.relative_to(ROOT)}  [{keep[p]}]")
    unclassified = [p for p in delete if family(p.name)[0] == "unclassified"]
    if unclassified:
        print("refusing: files outside both the keep closure and the discarded families:", file=sys.stderr)
        for p in unclassified:
            print(f"  {p.relative_to(ROOT)}", file=sys.stderr)
        return 2
    counts: dict[str, list[int]] = {}
    print(f"DELETE ({len(delete)}):")
    for p in delete:
        fam, why = family(p.name)
        size = p.stat().st_size
        counts.setdefault(fam, [0, 0])
        counts[fam][0] += 1
        counts[fam][1] += size
        print(f"  {p.relative_to(ROOT)}  {size} B  [{fam}: {why}]")
    print("per family:")
    for fam, (n, b) in sorted(counts.items()):
        print(f"  {fam}: {n} files, {b} B")
    print(f"total: {len(delete)} files, {sum(b for _, b in counts.values())} B")
    if args.execute:
        for p in delete:
            p.unlink()
        print(f"deleted {len(delete)} files")
    else:
        print("dry run; pass --execute to delete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
