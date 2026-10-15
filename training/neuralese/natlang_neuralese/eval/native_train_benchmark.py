"""Bounded whole-graph production-update profiling, using the existing trainer.

Diagnostic optimizer updates remain in this process and are not admitted as a
training run. Every selected graph is complete. Packing cases retain the actual
source prefix; no context or target truncation is applied. Weights and moments
continue between cases, so these timings support adaptive operations rather than
pretending to be frozen scientific comparisons.
"""
from __future__ import annotations
import argparse
import itertools
import json
import time
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex_cached
from ..model.native_model import load_native_model, load_native_checkpoint
from ..model.native_precision import PRODUCTION_NATIVE_LINEAR_PRECISIONS, set_native_linear_precision
from ..train.native_state_sft import (NativeGraphSource, batch_boundaries, make_native_optimizer,
                                     native_training_update)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parent = parser.add_mutually_exclusive_group(required=True)
    parent.add_argument('--base'); parent.add_argument('--checkpoint')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--graphs', type=Path, required=True)
    parser.add_argument('--pair-model')
    parser.add_argument('--cases', nargs='+', default=['graphs:1','graphs:2','graphs:4','graphs:8',
                                                       'tokens:8192','tokens:16384','tokens:32768'])
    parser.add_argument('--chunks', nargs='+', type=int, default=[128,1024])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--native-linear-precision', choices=PRODUCTION_NATIVE_LINEAR_PRECISIONS, default='ieee')
    parser.add_argument('--fp8-training', choices=('off','triton-experts','te-fp8'), default='off')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repeats < 2 or min(args.chunks) < 1:
        raise ValueError('profile at least two updates and positive whole-readout chunks')
    torch.manual_seed(711)
    if args.checkpoint:
        model, _, roles, config, _ = load_native_checkpoint(args.checkpoint, device='cuda')
    else:
        model, _, roles, config = load_native_model(args.base, dtype=getattr(torch,args.dtype), device='cuda')
    set_native_linear_precision(model,args.native_linear_precision)
    config['native_linear_precision'] = args.native_linear_precision
    source = NativeGraphSource(args.graphs,config['tokenizer_fingerprint'],required_split='train',
                               pair_model=args.pair_model,role_ids=roles.ids)
    cases=[]
    for case in args.cases:
        kind, raw = case.split(':'); size=int(raw)
        if kind not in {'graphs','tokens'} or size < 1:
            raise ValueError('case must be graphs:positive-count or tokens:positive-volume')
        first,last = batch_boundaries(source.entries,batch_graphs=size if kind=='graphs' else 1,
                                      batch_tokens=size if kind=='tokens' else 0)[0]
        cases.append((case,list(itertools.islice(source,first,last))))
    backend = None
    if args.fp8_training != 'off':
        from ..train.fp8_backend import make_backend
        backend=make_backend(args.fp8_training,model.port.hf,device='cuda')
        model.port.activation_checkpoint=backend.checkpoint
    model.train();model.port.checkpoint_layers=True
    optimizer,_=make_native_optimizer(model,lr=3e-6)
    report={'schema':'natlang.native-train-benchmark/1','source':config,'graphs':source.receipt(),
            'hardware':torch.cuda.get_device_name(),'torch':torch.__version__,'fp8':args.fp8_training,
            'optimizer':'shared-muon-sr-in-backward','checkpoint_layers':True,
            'whole_graphs':True,'updates_are_temporary_diagnostics':True,'cases':[]}
    package = Path(__file__).resolve().parents[1]
    report['code_files'] = {name: sha256_file_hex_cached(package/name) for name in (
        'model/shared_state.py','model/native_model.py','model/native_precision.py',
        'maple/fused_moe.py','train/native_state_sft.py','train/optim.py','eval/native_train_benchmark.py')}
    cumulative=0
    for name, batch in cases:
        for chunk in args.chunks:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            record={'packing':name,'chunk_tokens':chunk,'graphs':len(batch),
                    'graph_ids':[row.id for row in batch],
                    'forward_tokens':sum(row.plan.token_count for row in batch),
                    'supervised_tokens':sum(sum(w>0 for w in row.target_weights) for row in batch),
                    'updates':[]}
            record['status']='running'
            report['cases'].append(record)
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(report,indent=2)+'\n')
            try:
                for repeat in range(args.repeats):
                    torch.cuda.synchronize();start=time.monotonic()
                    loss,_=native_training_update(model,optimizer,batch,backend=backend,
                                                 checkpoint_layers=True,chunk_tokens=chunk)
                    torch.cuda.synchronize();elapsed=time.monotonic()-start;cumulative+=1
                    record['updates'].append({'repeat':repeat,'seconds':elapsed,
                        'loss':float(loss.detach()),'cumulative_diagnostic_updates':cumulative,
                        'supervised_tps':record['supervised_tokens']/elapsed})
            except torch.cuda.OutOfMemoryError as error:
                record['status']='out_of_memory'
                record['error']=str(error)
                args.out.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(record),flush=True)
                return 2
            record['status']='complete'
            record['cold_seconds']=record['updates'][0]['seconds']
            record['warm_seconds']=sum(v['seconds'] for v in record['updates'][1:])/(args.repeats-1)
            record['warm_supervised_tps']=record['supervised_tokens']/record['warm_seconds']
            record['case_peak_cuda_bytes']=torch.cuda.max_memory_allocated()
            record['cuda_allocated_bytes']=torch.cuda.memory_allocated()
            record['cuda_reserved_bytes']=torch.cuda.memory_reserved()
            args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(record),flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
