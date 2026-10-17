#!/usr/bin/env python3
"""Zero-update full-parent attention arithmetic comparison at actual source geometry.

Reuse the production capacity execution and state guards. Reference adjoints
spill individually to temporary storage, so the verifier never holds another
model's gradients in RAM. Timings exclude reference serialization/comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.neuralese_data.qualify_native_attention_index import diagnostic_optimizer, tensor_digest, gradient_name_aliases
from scripts.neuralese_data.qualify_native_causal_prefixes import gradient_metrics, metrics
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.eval.native_memory import qualify_capacity
from natlang_neuralese.model.native_precision import set_native_attention_precision


def paired_arithmetic_case(model, optimizer, example, *, candidate='tf32x3', atol, rtol,
                           checkpoint_sha256, source_binding, backend=None,
                           checkpoint_layers=True, chunk_tokens=1024, temporary_directory=None):
    """Compare the real output, loss and every observed named parameter gradient.

    The existing capacity verifier independently restores CPU/all-CUDA RNG,
    backend state, training modes and released gradients after each execution,
    and rejects any parameter or optimizer mutation. Restore arithmetic even
    after a mismatch or exception. Never arm an optimizer update here.
    """
    previous=getattr(model,'native_attention_precision','ieee')
    if previous==candidate:
        raise ValueError('attention comparison requires a different candidate arithmetic')
    if atol<=0 or rtol<=0 or not all(torch.isfinite(torch.tensor(x)) for x in (atol,rtol)):
        raise ValueError('attention comparison requires finite positive declared tolerances')
    inventory={name:{'shape':list(value.shape),'dtype':str(value.dtype)}
               for name,value in optimizer._native_named_parameters if value.requires_grad}
    names=gradient_name_aliases(model,optimizer)
    if not any(name.startswith('backbone.') for name in inventory):
        raise ValueError('attention arithmetic comparison requires actual production backbone trainables')
    comparisons={};reference={};seen=set();outputs={};geometries={};runs={}
    kwargs=dict(backend=backend,checkpoint_layers=checkpoint_layers,chunk_tokens=chunk_tokens,
                checkpoint_sha256=checkpoint_sha256,source_binding=source_binding)
    def execute(label,**extra):
        proof=qualify_capacity(model,optimizer,[example],**kwargs,**extra)
        if proof['passed'] and backend is not None:
            from natlang_neuralese.train.fp8_backend import require_fp8_projection_execution
            require_fp8_projection_execution(proof['executed_projections'],context='attention '+label)
        return proof
    try:
        with tempfile.TemporaryDirectory(prefix='native-attention-adjoints-',dir=temporary_directory) as directory:
            directory=Path(directory)
            def save_gradient(name,value):
                name=names[name]
                if name in reference:raise ValueError('attention reference repeats a named parameter adjoint')
                path=directory/(str(len(reference))+'.pt')
                tensor=value.detach().cpu().contiguous()
                torch.save(tensor,path)
                reference[name]={'path':path,'digest':tensor_digest(tensor)}
            def compare_gradient(name,value):
                name=names[name]
                if name not in reference or name in seen:
                    raise ValueError('attention candidate adds or repeats a named parameter adjoint')
                seen.add(name)
                expected=torch.load(reference[name]['path'],map_location=value.device,weights_only=True)
                if str(expected.dtype)!=str(value.dtype):
                    raise ValueError('attention arithmetic changed a gradient dtype')
                comparisons[name]=gradient_metrics(value,expected,atol=atol,rtol=rtol)
            for label,mode,observer in [('reference',previous,save_gradient),('candidate',candidate,compare_gradient)]:
                set_native_attention_precision(model,mode)
                def output(module,args,value):
                    if label in outputs:raise ValueError('attention comparison replayed a whole graph forward')
                    outputs[label]=value['h_final'].detach().cpu().clone()
                    plan=args[0]
                    geometries[label]={'forward_tokens':plan.token_count,'kv_slots':len(plan.slot_sources)}
                hook=model.register_forward_hook(output)
                try:
                    runs[label]=execute(label,gradient_observer=observer)
                finally:hook.remove()
                if not runs[label]['passed']:
                    return {'passed':False,'from':previous,'to':candidate,'capacity':runs,
                            'optimizer_updates':0,'task_quality_qualified':False}
            hidden=metrics(outputs['candidate'],outputs['reference'],atol=atol,rtol=rtol)
            losses=metrics(torch.tensor(runs['candidate']['cases'][0]['loss'],dtype=torch.float64),
                           torch.tensor(runs['reference']['cases'][0]['loss'],dtype=torch.float64),atol=atol,rtol=rtol)
            passed=(hidden['passed'] and losses['passed'] and seen==reference.keys()
                    and geometries['reference']==geometries['candidate']
                    and bool(comparisons) and all(value['passed'] for value in comparisons.values()))
            timings={}
            if passed:
                # Both kernels were warmed by the observed qualification above.
                # Keep serialization and FP64 comparison outside measured runs.
                for label,mode in [('reference',previous),('candidate',candidate)]:
                    set_native_attention_precision(model,mode)
                    proof=execute('timed '+label)
                    if not proof['passed']:raise ValueError('timed attention execution failed capacity checks')
                    timings[label]=proof['cases'][0]['seconds']
            return {'schema':'natlang.native-attention-arithmetic-paired/1','passed':passed,
                'from':previous,'to':candidate,'atol':atol,'rtol':rtol,
                'h_final':hidden,'loss':losses,'named_gradients':comparisons,
                'missing_gradient_names':sorted(reference.keys()-seen),
                'unobserved_trainable_names':sorted(inventory.keys()-reference.keys()),
                'production_trainable_inventory':inventory,
                'reference_gradient_digest':canonical_json_sha256_hex({name:row['digest'] for name,row in reference.items()}),
                'geometry':geometries,'seconds':timings,'capacity':runs,
                'retained_hidden_bytes':sum(value.numel()*value.element_size() for value in outputs.values()),
                'optimizer_updates':0,'task_quality_qualified':False,
                'timing_scope':'single warmed whole-graph forward/backward per mode; no serialization/comparison or SGD',
                'parent_checkpoint_sha256':checkpoint_sha256,'source':source_binding}
    finally:
        set_native_attention_precision(model,previous)


def main(argv=None):
    import json
    from scripts.neuralese_data.qualify_native_causal_prefixes import load_qualification_model
    from natlang_neuralese.common.checkpoint_offload import configure_checkpoint_input_offload
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.eval.native_memory import capacity_identity
    from natlang_neuralese.model import shared_state
    from natlang_neuralese.train.fp8_backend import make_backend
    from natlang_neuralese.train.native_resident import binding,bound,code_inventory
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('base','checkpoint','source','package','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--capacity-certificate',required=True,help='exact JSON path/SHA binding')
    parser.add_argument('--model',choices=['lfm','mellum'],required=True)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--candidate',choices=['tf32x3'],default='tf32x3')
    parser.add_argument('--atol',type=float,required=True)
    parser.add_argument('--rtol',type=float,required=True)
    parser.add_argument('--temporary-directory',type=Path,required=True)
    parser.add_argument('--native-participant')
    args=parser.parse_args(argv)
    if Path(shared_state.__file__).resolve()!=(args.package/'model/shared_state.py').resolve():
        raise ValueError('attention comparison imported a different numerical runtime')
    certificate=json.loads(bound(json.loads(args.capacity_certificate)).read_text())
    if certificate.get('passed') is not True:
        raise ValueError('attention comparison requires an actually passed source capacity certificate')
    identity=certificate['identity'];args.linear_precision=identity['model_config']['native_linear_precision']
    model,tokenizer,roles,config,parent,options=load_qualification_model(args)
    backend=None
    if args.model=='mellum':
        backend=make_backend(identity['fp8']['kind'],model.port.hf,device='cuda',
            recipe=identity['fp8']['recipe'],amax_history_len=identity['fp8']['amax_history_len'])
        if options.get('extra_state') is not None:backend.restore_extra_state(options['extra_state'])
        model.port.activation_checkpoint=backend.checkpoint
    configure_checkpoint_input_offload(model.port,enabled=identity['checkpoint_input_offload'],
        directory=identity.get('checkpoint_input_offload_directory'))
    model.share_causal_prefixes=identity.get('share_causal_prefixes',False)
    actual_args=argparse.Namespace(**{key:identity[key] for key in
        ('batch_graphs','batch_tokens','checkpoint_layers','chunk_tokens','checkpoint_input_offload')},
        checkpoint_input_offload_directory=identity.get('checkpoint_input_offload_directory'),
        device='cuda',share_causal_prefixes=model.share_causal_prefixes,fp8_training=identity['fp8']['kind'],
        fp8_recipe=identity['fp8']['recipe'],fp8_amax_history_len=identity['fp8']['amax_history_len'])
    if capacity_identity(config,actual_args,backend)!=identity:
        raise ValueError('actual restored parent/backend/geometry differs from bound capacity identity')
    source=NativeGraphSource(args.source,required_split='train',tokenizer_fingerprint=config['tokenizer_fingerprint'],
        role_ids=config['marker_ids'])
    example,=source.iter_indices([args.index])
    args.temporary_directory.mkdir(parents=True,exist_ok=True)
    result=paired_arithmetic_case(model,diagnostic_optimizer(model),example,candidate=args.candidate,
        atol=args.atol,rtol=args.rtol,checkpoint_sha256=parent['sha256'],source_binding=binding(args.source),
        backend=backend,checkpoint_layers=identity['checkpoint_layers'],chunk_tokens=identity['chunk_tokens'],
        temporary_directory=args.temporary_directory)
    candidate_identity={**identity,'model_config':{**config,'native_attention_precision':args.candidate}}
    result.update(reference_identity=identity,candidate_identity=candidate_identity,
        reference_capacity_certificate=binding(bound(json.loads(args.capacity_certificate))),
        runtime_code_sha256=canonical_json_sha256_hex(code_inventory(args.package)),
        entrypoint=binding(Path(__file__)),source_row={key:source.entries[args.index][key] for key in ('id','binding')},
        scope='independent read-only parent comparison; no training admission, SGD, optimizer/RNG/exposure file modification')
    write_json_durable(args.output,result)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
