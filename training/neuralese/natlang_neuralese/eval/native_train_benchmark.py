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
                                     native_training_update, packed_graph_loss)


def read_only_comparison(model, batch, *, layouts=((True, 1024), (False, 2048)),
                         repeats=3, gradient_atol=1e-5, loss_atol=1e-4):
    """Compare complete production-policy gradients without taking an update.

    Callers configure the actual production parameter policy with
    make_native_optimizer(in_backward=False). No optimizer is consumed here.
    Each repetition starts with the same RNG; the caller's RNG is restored.
    """
    from ..train.loop import capture_training_rng_state, restore_training_rng_state
    if repeats < 2 or not layouts or any(chunk < 1 for _, chunk in layouts):
        raise ValueError('need a baseline, positive chunks and at least two repetitions')
    device = next(model.parameters()).device
    rng = capture_training_rng_state(device)
    training, checkpoint_layers = model.training, model.port.checkpoint_layers
    reference, baseline_loss, records = None, None, []
    supervised_tokens = sum(sum(weight > 0 for weight in row.target_weights) for row in batch)
    def sync():
        if device.type == 'cuda': torch.cuda.synchronize(device)
    try:
        model.train()
        for checkpointed, chunk in layouts:
            model.zero_grad(set_to_none=True)
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            record = {'checkpoint_layers': checkpointed, 'chunk_tokens': chunk,
                      'optimizer_updates': 0, 'iterations': [], 'status': 'running'}
            records.append(record)
            try:
                for repeat in range(repeats):
                    restore_training_rng_state(rng, device)
                    model.zero_grad(set_to_none=True)
                    sync(); start = time.monotonic()
                    loss, _ = packed_graph_loss(model, batch,
                        checkpoint_layers=checkpointed, chunk_tokens=chunk)
                    loss.backward(); sync()
                    record['iterations'].append({'seconds': time.monotonic()-start,
                                                 'loss': float(loss.detach())})
                gradients = {name: value.grad.detach().cpu().clone()
                    for name, value in model.named_parameters() if value.grad is not None}
                if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
                    raise FloatingPointError('missing or nonfinite actual production gradients')
                actual_loss = record['iterations'][-1]['loss']
                if reference is None:
                    reference, baseline_loss = gradients, actual_loss
                    gap = 0.
                else:
                    if gradients.keys() != reference.keys():
                        raise ValueError('checkpoint/chunk comparison changed gradient membership')
                    gap = max(float((g.float()-reference[name].float()).abs().max())
                              for name, g in gradients.items())
                record.update(gradient_max_abs=gap, loss_abs=abs(actual_loss-baseline_loss),
                    gradient_parameters=len(gradients),
                    warm_seconds=sum(x['seconds'] for x in record['iterations'][1:])/(repeats-1),
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
                record['warm_supervised_tps'] = supervised_tokens / record['warm_seconds']
                record['status'] = ('passed' if gap <= gradient_atol and
                    record['loss_abs'] <= loss_atol else 'numerical_mismatch')
                del gradients
            except torch.cuda.OutOfMemoryError as error:
                record.update(status='out_of_memory', error=str(error))
                model.zero_grad(set_to_none=True)
                if device.type == 'cuda': torch.cuda.empty_cache()
                break
        return {'schema': 'natlang.native-read-only-training-comparison/1',
                'optimizer_updates': 0, 'rng_restored': True, 'whole_graphs': True,
                'gradient_atol': gradient_atol, 'loss_atol': loss_atol,
                'forward_tokens': sum(row.plan.token_count for row in batch),
                'supervised_tokens': supervised_tokens,
                'graph_ids': [row.id for row in batch], 'cases': records}
    finally:
        model.zero_grad(set_to_none=True)
        model.train(training); model.port.checkpoint_layers = checkpoint_layers
        restore_training_rng_state(rng, device)


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
    parser.add_argument('--read-only', action='store_true',
        help='Compare checkpoint1024/direct2048 full gradients without optimizer updates')
    parser.add_argument('--start-index', type=int, default=0)
    args = parser.parse_args(argv)
    if args.repeats < 2 or min(args.chunks) < 1:
        raise ValueError('profile at least two updates and positive whole-readout chunks')
    torch.manual_seed(711)
    if args.checkpoint:
        model, _, roles, config, parent_state = load_native_checkpoint(args.checkpoint, device='cuda')
    else:
        model, _, roles, config = load_native_model(args.base, dtype=getattr(torch,args.dtype), device='cuda')
    if args.read_only and (not args.checkpoint or args.fp8_training != 'off'):
        raise ValueError('read-only comparison requires an actual checkpoint and FP8 off')
    if not args.read_only:
        set_native_linear_precision(model,args.native_linear_precision)
        config['native_linear_precision'] = args.native_linear_precision
    source = NativeGraphSource(args.graphs,config['tokenizer_fingerprint'],required_split='train',
                               pair_model=args.pair_model,role_ids=roles.ids)
    cases=[]
    for case in args.cases:
        kind, raw = case.split(':'); size=int(raw)
        if kind not in {'graphs','tokens'} or size < 1:
            raise ValueError('case must be graphs:positive-count or tokens:positive-volume')
        if not 0 <= args.start_index < len(source): raise ValueError('start index outside source')
        first,last = batch_boundaries(source.entries[args.start_index:],batch_graphs=size if kind=='graphs' else 1,
                                      batch_tokens=size if kind=='tokens' else 0)[0]
        first += args.start_index; last += args.start_index
        cases.append((case,list(itertools.islice(source,first,last))))
    if args.read_only:
        if len(cases) != 1: raise ValueError('read-only comparison uses one complete packed batch')
        optimizer, _ = make_native_optimizer(model, lr=3e-6, in_backward=False)
        from ..train.loop import restore_training_rng_state
        restore_training_rng_state(parent_state['training']['rng'], 'cuda')
        report = read_only_comparison(model, cases[0][1], repeats=args.repeats)
        report.update(checkpoint={'path':args.checkpoint,'sha256':sha256_file_hex_cached(args.checkpoint)},
                      model_config=config, hardware=torch.cuda.get_device_name(), torch=torch.__version__)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        return 0 if all(row['status']=='passed' for row in report['cases']) else 2
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
