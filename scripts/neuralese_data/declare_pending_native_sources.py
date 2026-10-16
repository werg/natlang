"""Compile immutable admitted long-row obligations using the shared native parser."""
from __future__ import annotations
import argparse
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.native_pending_sources import declaration, load_pending_sources
from scripts.neuralese_data.build_acquired_full_session_graphs import prepare_tokenizer
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-cohort',type=Path,required=True)
    parser.add_argument('--pending-native-graphs',type=Path,required=True)
    parser.add_argument('--base-tokenizer',type=Path,required=True)
    parser.add_argument('--required-visits',type=int,default=1)
    parser.add_argument('--evidence',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    prepared=prepare_tokenizer(args.base_tokenizer)
    value=declaration(args.full_cohort,args.pending_native_graphs,
        tokenizer_fingerprint=prepared.expanded_fingerprint,role_ids=provision(prepared.tokenizer),
        required_visits=args.required_visits,output_base=args.output.parent,evidence=[{'path':str(path.resolve(strict=True)),
            'sha256':sha256_file_hex_cached(path)} for path in args.evidence])
    prepared.validate(args.base_tokenizer)
    if args.output.exists():
        import json
        if json.loads(args.output.read_text()) != value:
            raise ValueError('pending native declaration output already binds different immutable sources')
    else:
        write_json_durable(args.output,value)
    load_pending_sources(args.output,tokenizer_fingerprint=prepared.expanded_fingerprint)
    return 0


if __name__=='__main__':raise SystemExit(main())
