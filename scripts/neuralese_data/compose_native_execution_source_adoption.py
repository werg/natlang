#!/usr/bin/env python3
"""Compose an earned execution selection with a verified zero-SGD source adoption."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('selected-execution','selected-execution-sha256','source-adoption','source-adoption-sha256','output'):
        parser.add_argument('--'+flag,required=True)
    args=parser.parse_args()
    from natlang_neuralese.train.native_execution_transition import publish_source_adopted_selection
    result=publish_source_adopted_selection(
        {'path':str(Path(args.selected_execution).resolve(strict=True)),'sha256':args.selected_execution_sha256},
        {'path':str(Path(args.source_adoption).resolve(strict=True)),'sha256':args.source_adoption_sha256},
        out=args.output)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
