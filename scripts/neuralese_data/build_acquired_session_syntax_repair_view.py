#!/usr/bin/env python3
"""Build an immutable child export for strictly recognized source syntax repairs."""
from __future__ import annotations

import argparse
import json

from scripts.neuralese_data.acquired_session_syntax_repair import build_repair_view


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-export', required=True)
    parser.add_argument('--candidate-inventory', required=True)
    parser.add_argument('--pending-audit', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = build_repair_view(parent_export_path=args.parent_export,
        candidate_audit_path=args.candidate_inventory,
        pending_audit_path=args.pending_audit, output_dir=args.out)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
