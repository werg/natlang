#!/usr/bin/env python3
"""Start or resume the exact reviewed dispatcher through one restartable command."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    plan = args.plan.resolve(strict=True)
    declaration = json.loads(plan.read_text())
    dispatcher = Path(__file__).with_name('run_reviewed_luna_dispatcher.py')
    command = [sys.executable, str(dispatcher), str(plan), '--sha256', args.sha256]
    # The dispatcher owns validation, leases, actor identities and the journal.
    # A guardian retry must reevaluate this after the first process created it.
    if Path(declaration['claim_ledger']).exists():
        command.append('--resume')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
