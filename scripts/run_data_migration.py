#!/usr/bin/env python3
"""Preview a natlang-planned SQLite import, optionally apply it transactionally."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from applications.data_migration import Export, MigrationStudio
from applications.teacher import teacher_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSON document with an exports array")
    parser.add_argument("target", type=Path, help="SQLite target database")
    parser.add_argument("out", type=Path, help="new preview/result JSON file")
    parser.add_argument("--apply", action="store_true", help="commit this preview in one transaction")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    trace_dir = args.out.parent / (args.out.stem + "-traces")
    result_path = args.out.with_suffix(".applied.json")
    if args.out.exists() or trace_dir.exists() or (args.apply and result_path.exists()):
        parser.error("output, result or trace directory already exists")
    document = json.loads(args.source.read_text())
    exports = [Export(row["source"], tuple(row["customers"]), tuple(row["orders"]))
               for row in document["exports"]]
    studio = MigrationStudio(args.target,
                             agent_factory=teacher_factory(args.server),
                             model_id=args.model_id, root_seed=args.seed, trace_dir=trace_dir)
    preview = studio.preview(exports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(preview, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if args.apply:
        result = studio.apply(preview, exports)
        with result_path.open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(f"{preview['counts']}; {'applied' if args.apply else 'preview only'}: {args.out}")


if __name__ == "__main__":
    main()
