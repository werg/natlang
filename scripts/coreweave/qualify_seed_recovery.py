#!/usr/bin/env python3
"""Decode and verify an independently downloaded full seed in its own ledger unit."""
import argparse
from pathlib import Path
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path);parser.add_argument('sha256');parser.add_argument('bytes',type=int)
    parser.add_argument('step',type=int);parser.add_argument('out',type=Path);args=parser.parse_args()
    if args.checkpoint.stat().st_size!=args.bytes or sha256_file_hex(args.checkpoint,drop_cache=True)!=args.sha256:
        raise ValueError('independent seed copy differs from exact size/hash')
    state=CheckpointPolicy.load(args.checkpoint);training=state.get('training') or {}
    if (state.get('schema')!=CHECKPOINT_SCHEMA or training.get('schema')!=TRAINING_SCHEMA
            or state.get('step')!=args.step or not state.get('model_config')
            or not state.get('backbone_state') or not state.get('interface_state')
            or training.get('global_step')!=args.step or not training.get('optimizer')
            or not training.get('rng') or not training.get('exposure_counts') or not training.get('source_receipts')):
        raise ValueError('independent seed copy lacks its exact full optimizer/RNG/exposure/source state')
    result={'schema':'natlang.lfm3-seed-independent-full-state/1','checkpoint':str(args.checkpoint.resolve()),
            'sha256':args.sha256,'bytes':args.bytes,'global_step':args.step,'passed':True}
    if args.out.exists():raise ValueError('seed verification receipt already exists')
    args.out.parent.mkdir(parents=True,exist_ok=True);write_json_durable(args.out,result)


if __name__=='__main__':main()
