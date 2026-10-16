"""Stream and durably bind one complete acquired-source protection closure."""
from __future__ import annotations

import argparse
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.prepare_acquired_session_admission import verify_global_source_closure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('source closure receipts are immutable; choose a new output path')
    protection = current_protection(args.protection)
    accounting = verify_global_source_closure(args.protection, protection,
                                              write_receipt_path=args.output)
    print(f'{args.output} {sha256_file_hex(args.output)} {accounting["normalized_sessions"]}')


if __name__ == '__main__':
    main()
