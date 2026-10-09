#!/usr/bin/env python3
"""The `view`/`ask` summarizer-family corpus (plans/neuralese/VIEW_CORPUS.md).

Subcommands:
  fetch   download the pinned public-dataset slices to the raw directory (network; run under the memory ledger)
  build   build port records from the raw directory and local caches, close splits, validate, write a manifest

Example (under the ledger):
  python3 scripts/memory_ledger.py run --unit natlang-view-corpus-HHMM --budget-gb 8 --class experiment --wait 600 \
    --workdir /home/werg/natlang -- sh -c ".venv-neuralese/bin/python scripts/neuralese_view_corpus.py build \
    --out /home/werg/data/natlang-corpora/view-corpus-slice-20261009-v1 > /tmp/vc.log 2>&1; echo EXIT \\$? >> /tmp/vc.log"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from neuralese_data import view_corpus, view_sources  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--raw", type=Path, default=view_sources.DEFAULT_RAW)
    f.add_argument("--csn-train-rows", type=int, default=20000)
    f.add_argument("--websrc-pages-per-site", type=int, default=12)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, default=view_sources.DEFAULT_RAW)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--corpus-id", default=None, help="defaults to the output directory name")
    b.add_argument("--caps", default=None, help="JSON object overriding per-source document caps (see view_corpus.DEFAULT_CAPS)")
    b.add_argument("--max-source-chars", type=int, default=view_corpus.MAX_SOURCE_CHARS)
    b.add_argument("--min-source-chars", type=int, default=view_corpus.MIN_SOURCE_CHARS)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--protected", type=Path, default=view_corpus.PROTECTED_INDEX)
    b.add_argument("--only", action="append", default=None, help="restrict to these source adapters")
    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        report = view_sources.fetch(args.raw, csn_train_rows=args.csn_train_rows,
                                    websrc_pages_per_site=args.websrc_pages_per_site)
        print(json.dumps({"files": len(report["files"]), "raw": report["raw"]}))
        return 0
    caps = dict(view_corpus.DEFAULT_CAPS)
    if args.caps:
        caps.update(json.loads(args.caps))
    manifest = view_corpus.build(args.raw, args.out, corpus_id=args.corpus_id or args.out.name, caps=caps,
                                 max_source_chars=args.max_source_chars, min_source_chars=args.min_source_chars,
                                 seed=args.seed, protected_path=args.protected, only=args.only)
    print(json.dumps({k: manifest[k] for k in ("records", "by_artifact", "by_task", "by_split")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
