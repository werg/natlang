#!/usr/bin/env python3
"""Stage or publish a verified full-state source correction with zero SGD."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='operation',required=True)
    stage=sub.add_parser('prepare')
    for flag in ('parent','parent-sha256','prior-recipe','successor-recipe','stage-id','output'):
        stage.add_argument('--'+flag,required=True)
    publish=sub.add_parser('publish')
    for flag in ('state','proof','proof-sha256','output'):
        publish.add_argument('--'+flag,required=True)
    args=parser.parse_args()
    if args.operation=='prepare':
        from natlang_neuralese.train.native_zero_sgd_adoption import prepare
        result=prepare({'path':str(Path(args.parent).resolve(strict=True)),'sha256':args.parent_sha256},
            prior_recipe=args.prior_recipe,successor_recipe=args.successor_recipe,
            stage_id=args.stage_id,output=args.output)
    else:
        from natlang_neuralese.rl.native_direct_sft_recovery import publish_zero_sgd_adoption
        result=publish_zero_sgd_adoption(args.state,proof_ref={'path':str(Path(args.proof).resolve(strict=True)),
            'sha256':args.proof_sha256},out=args.output)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
