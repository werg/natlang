#!/usr/bin/env python3
"""Reject unverified continuation data before preparing an improvement round."""
import argparse
from pathlib import Path

from create_improvement_round import validate_correction_turns
from run_training_pipeline import atomic_json, digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = {'version': 'natlang.correction_validation/1', 'input_sha256': digest_file(args.input),
              'approved_decisions': validate_correction_turns(args.input)}
    if args.output.exists():
        import json
        if json.loads(args.output.read_text()) != report:
            raise ValueError('refusing to replace changed correction validation')
    else:
        atomic_json(args.output, report)
    print(report)


if __name__ == '__main__':
    main()
