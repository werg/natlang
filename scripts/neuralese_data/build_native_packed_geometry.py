#!/usr/bin/env python3
"""Explicit CPU preparation of exact native packed-batch source geometry."""
import argparse
import fcntl
import json
import signal
import time
from pathlib import Path

from natlang_neuralese.train.native_packed_capacity import geometry_for_source, index_source
from natlang_neuralese.train.native_resident import binding
from natlang_neuralese.common.jsonio import write_json_durable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--tokenizer-fingerprint',required=True)
    parser.add_argument('--role-ids',type=int,nargs=4,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prior',help='exact JSON path/SHA binding of a verified prior geometry sidecar')
    parser.add_argument('--checkpoint-rows',type=int,default=10000,
        help='persist a verified prefix after this many additional rows; zero disables periodic saves')
    args=parser.parse_args()
    if args.checkpoint_rows<0:parser.error('--checkpoint-rows must be nonnegative')
    started=time.monotonic()
    args.out.mkdir(parents=True,exist_ok=True)
    query={'source':binding(args.source),'tokenizer':args.tokenizer_fingerprint,'roles':args.role_ids,
        'initial_prior':json.loads(args.prior) if args.prior else None}
    state_path=args.out/'packed-geometry-build.json'
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    with (args.out/'packed-geometry-build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=query['initial_prior']
        if state_path.exists():
            state=json.loads(state_path.read_text())
            if state.get('query')!=query:raise ValueError('packed geometry build changed its exact source or arguments')
            prior=state['geometry']
        def progress(reference,count,total):
            record={'query':query,'geometry':reference,'rows':count,'total_rows':total,'complete':count==total,
                'seconds_this_attempt':time.monotonic()-started,'optimizer_updates':0,'training_exposure':0}
            write_json_durable(state_path,record)
            print(json.dumps(record),flush=True)
        source,indexed=index_source(args.source,args.tokenizer_fingerprint,args.role_ids)
        _,reference,scan=geometry_for_source(source,args.out,prior=prior,
            allow_build=True,indexed_geometry=indexed,checkpoint_rows=args.checkpoint_rows,progress_callback=progress)
    print(json.dumps({'geometry':reference,**scan,'seconds':time.monotonic()-started,
                      'optimizer_updates':0,'training_exposure':0}),flush=True)


if __name__=='__main__':main()
