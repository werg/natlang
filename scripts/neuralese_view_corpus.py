#!/usr/bin/env python3
"""The `view`/`ask` summarizer-family corpus (plans/neuralese/VIEW_CORPUS.md).

Subcommands:
  fetch   download the pinned public-dataset slices to the raw directory (network; run under the memory ledger)
  build   build port records from the raw directory and local caches, close splits (also against published
          corpora: --cross-index, see neuralese_data/cross_corpus.py), validate, write a manifest

v2 (view-ask-20261010-v2): fetch --v2 into a new raw directory, then build --preset v2 --cross-index cross-corpus-index-s1-full-final-20261003-v1
--cross-index cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v3-v1 (VIEW_CORPUS.md §4 and §5).
v3 (view-ask-20261010-v3): a hard-linked copy of the v2 raw directory, fetch --v3 into it, then build --preset v3 with
the same --cross-index arguments (VIEW_CORPUS.md §5.3). Licences are recorded per record as provenance facts; no
record is omitted for its licence (owner rule 2026-10-10).
v4 (view-ask-20261010-v4): v3's raw directory and caps (--preset v4), SWE-rebench tool-output splits from the harness
bench's placements (S1-aligned), closed with --cross-index cross-corpus-index-s1-full-final-20261003-v1 --cross-index
cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1 (VIEW_CORPUS.md §5.4).

Example (under the ledger; <repo> and <data_nvme> are natlang_neuralese.common.paths roots):
  python3 scripts/memory_ledger.py run --unit natlang-view-corpus-HHMM --budget-gb 8 --class experiment --wait 600 \
    --workdir <repo> -- sh -c ".venv-neuralese/bin/python scripts/neuralese_view_corpus.py build \
    --out <data_nvme>/natlang-corpora/view-corpus-slice-20261009-v1 > /tmp/vc.log 2>&1; echo EXIT \\$? >> /tmp/vc.log"
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
    f.add_argument("--v2", action="store_true", help="v2 slices: 80k Python rows, 15k rows of each other CodeSearchNet "
                   "language, licence files, 40 WebSRC pages per site, TabFact tables")
    f.add_argument("--v3", action="store_true", help="the v2 slices plus BookSum, XSum, GovReport, Mind2Web (5 train "
                   "files), LogHub and a Common Crawl WARC prefix")
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
    b.add_argument("--preset", choices=("v1", "v2", "v3", "v4"), default="v1",
                   help="caps preset (v2 adds the v2 sources, v3 the v3 sources; v4 = v3's caps)")
    b.add_argument("--cross-index", action="append", default=[],
                   help="published-corpus index: a directory (cross_corpus.index_corpus) or a registered "
                        "cross-corpus-index id (neuralese_data/cross_corpus_registry.py); repeatable")
    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        if args.v2 or args.v3:
            report = view_sources.fetch(args.raw, csn_train_rows=80000, websrc_pages_per_site=40,
                                        csn_languages=view_sources.CSN_LANGUAGES, csn_lang_train_rows=15000,
                                        csn_licenses=True, tabfact_tables={"train": 2600, "validation": 160, "test": 160},
                                        v3=args.v3)
        else:
            report = view_sources.fetch(args.raw, csn_train_rows=args.csn_train_rows,
                                        websrc_pages_per_site=args.websrc_pages_per_site)
        print(json.dumps({"files": len(report["files"]), "raw": report["raw"]}))
        return 0
    caps = dict({"v1": view_corpus.DEFAULT_CAPS, "v2": view_corpus.V2_CAPS, "v3": view_corpus.V3_CAPS, "v4": view_corpus.V3_CAPS}[args.preset])
    if args.caps:
        caps.update(json.loads(args.caps))
    manifest = view_corpus.build(args.raw, args.out, corpus_id=args.corpus_id or args.out.name, caps=caps,
                                 max_source_chars=args.max_source_chars, min_source_chars=args.min_source_chars,
                                 seed=args.seed, protected_path=args.protected, only=args.only,
                                 cross_indexes=args.cross_index, log=lambda m: print(m, flush=True))
    print(json.dumps({k: manifest[k] for k in ("records", "by_artifact", "by_task", "by_split")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
