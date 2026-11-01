"""CUDA-only diagnostic for the real two-update primal-sharing contract; no tolerances are changed."""
import argparse,copy,json
from pathlib import Path
import torch
from test_shared_state import tiny
from test_native_communication_prefix import episodes
from test_native_prefix_packing import examples as ordinary_examples
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.model.native_parameter_precision import configure_norm_precision
from natlang_neuralese.model.native_precision import set_native_linear_precision
from natlang_neuralese.train.backbone_policy import configure_backbone_training
from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update


def differences(a,b,path=''):
    if isinstance(a,torch.Tensor):
        if a.shape!=b.shape or a.dtype!=b.dtype:return [{'path':path,'shape_or_dtype_mismatch':True}]
        left=a.detach().cpu();right=b.detach().cpu();neq=left!=right
        if not neq.any():return []
        x=left.float();y=right.float();delta=(x-y).abs();r={'path':path,'dtype':str(a.dtype),'count':int(neq.sum()),'elements':a.numel(),
          'max_abs':float(delta.max()),'relative_l2':float(delta.norm()/y.norm().clamp_min(1e-30))}
        if a.dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
            r['adjacent'] = int((neq & (torch.nextafter(right,left)==left)).sum())
        return [r]
    if isinstance(a,dict):
        if a.keys()!=b.keys():return [{'path':path,'keys_a':sorted(map(str,a.keys())),'keys_b':sorted(map(str,b.keys()))}]
        return [d for key in a for d in differences(a[key],b[key],path+'/'+str(key))]
    if isinstance(a,(tuple,list)):
        if len(a)!=len(b):return [{'path':path,'length_mismatch':[len(a),len(b)]}]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+'/'+str(i))]
    return [] if a==b else [{'path':path,'a':str(a),'b':str(b)}]


def run(kind):
    model=SharedStateModel(tiny('lfm').to(device='cuda',dtype=torch.bfloat16),(58,59,60,61),rank=4,native_norm_execution='stable-v1')
    configure_norm_precision(model,'fp32-norms-v1');set_native_linear_precision(model,'bf16x6');configure_backbone_training(model.port,'latent')
    with torch.no_grad():model.adapter.up.weight.normal_(std=.1)
    donor=copy.deepcopy(model);model.share_causal_prefixes=True
    events=[{},{}];handles=[]
    for which,owner in enumerate((donor,model)):
        for name,param in owner.named_parameters():
            if not param.requires_grad:continue
            def capture(value,*,name=name,which=which):
                if value.grad is not None:events[which].setdefault(name,[]).append(value.grad.detach().cpu().clone())
            handles.append(param.register_post_accumulate_grad_hook(capture))
    opt_a,_=make_native_optimizer(donor,lr=1e-5,in_backward=True);opt_b,_=make_native_optimizer(model,lr=1e-5,in_backward=True)
    rows=episodes(empty_relay=True) if kind=='communication' else ordinary_examples();steps=[]
    for step in range(2):
        events[0].clear();events[1].clear()
        cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
        expected,ref=native_training_update(donor,opt_a,rows,chunk_tokens=3)
        cpu_after=torch.get_rng_state();gpu_after=torch.cuda.get_rng_state()
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
        actual,shared=native_training_update(model,opt_b,rows,chunk_tokens=3)
        steps.append({'step':step+1,'loss':differences(actual,expected,'loss'),
          'weights':differences(model.state_dict(),donor.state_dict(),'weights'),
          'optimizer':differences(opt_b.state_dict(),opt_a.state_dict(),'optimizer'),
          'post_accumulate_events':differences(events[1],events[0],'gradients'),
          'cpu_rng_equal':torch.equal(torch.get_rng_state(),cpu_after),
          'cuda_rng_equal':torch.equal(torch.cuda.get_rng_state(),gpu_after),
          'physical_tokens':[[item.get('packed_physical_tokens') for item in details]
                             for details in (ref,shared)]})
    for h in handles:h.remove()
    return steps


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; request root allocation')
    torch.manual_seed(18813);torch.cuda.manual_seed_all(18813)
    result={'schema':'natlang.primal-update-diagnostic/1','cases':{}}
    try:
        for kind in ('communication','ordinary'):
            try: result['cases'][kind]={'steps':run(kind)}
            except Exception as error:result['cases'][kind]={'error':repr(error)}
            Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    finally:
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k: ('error' if 'error' in v else 'recorded') for k,v in result['cases'].items()}))

if __name__=='__main__':main()
