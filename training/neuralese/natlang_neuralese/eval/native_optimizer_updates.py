"""Nonmutating actual optimizer consequences of complete communication VJPs.

This observes the existing full gradient comparison. It is diagnostic evidence,
never a replacement gradient/qualification gate or a production optimizer.
"""
from __future__ import annotations
import copy
import os
import torch
from .native_gradients import gradient_metrics
from ..train.optim import PortMuonAdamW,MuonSR

SCHEMA='natlang.native-complete-communication-optimizer-comparison/1'


def _copy(value):
    if torch.is_tensor(value):return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value,dict):return {key:_copy(item)for key,item in value.items()}
    if isinstance(value,list):return [_copy(item)for item in value]
    if isinstance(value,tuple):return tuple(_copy(item)for item in value)
    return copy.deepcopy(value)


def _signature(value):
    if torch.is_tensor(value):return ('tensor',id(value),value._version,str(value.dtype),tuple(value.shape))
    if isinstance(value,dict):return tuple((key,_signature(item))for key,item in value.items())
    if isinstance(value,(list,tuple)):return (type(value).__name__,tuple(_signature(item)for item in value))
    return (type(value).__name__,value)


def _metadata(value):
    if torch.is_tensor(value):
        return {'dtype':str(value.dtype),'shape':list(value.shape),'value':value.detach().cpu().tolist()}
    if isinstance(value,dict):return {key:_metadata(item)for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_metadata(item)for item in value]
    if type(value)in(str,int,float,bool)or value is None:return value
    raise ValueError('unsupported production optimizer group metadata')


def _bytes(value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(_bytes(item)for item in value.values())
    if isinstance(value,(list,tuple)):return sum(_bytes(item)for item in value)
    return 0


class CompleteOptimizerComparison:
    """A callable observer for ``compare_complete_gradients``.

    Every shadow is the source's actual child class, group, parameter dtype and
    inherited state. One parameter pair at a time bounds additional storage.
    The source optimizer is never stepped or loaded and no hook is registered.
    """
    def __init__(self,optimizer,named,*,gradient_atol=1e-5,gradient_rtol=1e-4,
                 update_atol=0.,update_rtol=1e-4):
        if not isinstance(optimizer,PortMuonAdamW):
            raise ValueError('complete update diagnosis needs the actual production PortMuonAdamW')
        named=list(named)
        self.named=dict(named);self.completed=None;self.buffer=None;self.buffer_bytes=0;self.aborted=False
        if len(self.named)!=len(named)or len({id(p)for _,p in named})!=len(named):
            raise ValueError('complete update diagnosis needs unique actual parameter names')
        self.optimizer=optimizer;self.slots={};self.comparisons={};self.order=[];self.largest_pair_bytes=0
        self.gradient_atol,self.gradient_rtol=gradient_atol,gradient_rtol
        self.update_atol,self.update_rtol=update_atol,update_rtol
        for value in(gradient_atol,gradient_rtol,update_atol,update_rtol):
            if type(value)not in(int,float)or not 0<=value<float('inf'):
                raise ValueError('complete update diagnosis needs finite nonnegative gates')
        by_id={id(p):name for name,p in named}
        supported=(torch.optim.AdamW,torch.optim.Muon,MuonSR)
        for kind,child in(('muon',optimizer.muon),('adamw',optimizer.auxiliary),(optimizer.latent_key,optimizer.latent)):
            if child is None:continue
            if type(child)not in supported or getattr(child,'_armed',None)is not None:
                raise ValueError('unsupported or currently armed production child optimizer')
            for group in child.param_groups:
                for index,p in enumerate(group['params']):
                    name=by_id.get(id(p))
                    if name is None or name in self.slots:
                        raise ValueError('optimizer and complete parameter inventory differ')
                    self.slots[name]=(kind,child,group,index,p)
        if set(self.slots)!=set(self.named):raise ValueError('complete update diagnosis omitted an optimizer parameter')
        self.devices=sorted({p.device.index for _,p in named if p.device.type=='cuda'})
        self.private_rng=self._capture_rng()
        self.source_signature=self._source_signature()

    def _capture_rng(self):
        return {'cpu':torch.get_rng_state().clone(),
                'cuda':{device:torch.cuda.get_rng_state(device).clone()for device in self.devices}}

    @staticmethod
    def _restore_rng(state):
        torch.set_rng_state(state['cpu'])
        for device,value in state['cuda'].items():torch.cuda.set_rng_state(value,device)

    def _source_signature(self):
        return (_signature(self.optimizer.state_dict()),tuple((name,id(p),p._version,
            None if p.grad is None else(id(p.grad),p.grad._version))for name,p in self.named.items()))

    def assert_preserved(self):
        if self._source_signature()!=self.source_signature:
            raise RuntimeError('complete update observer modified source optimizer, parameter or existing gradient')

    def _shadow(self,name,gradient):
        kind,child,group,index,p=self.slots[name]
        parameter=torch.nn.Parameter(p.detach().clone(memory_format=torch.preserve_format))
        # Follow PyTorch AccumulateGrad's ordinary parameter-layout contract.
        parameter.grad=torch.empty_like(parameter,memory_format=torch.preserve_format)
        parameter.grad.copy_(gradient)
        selected={key:_copy(value)for key,value in group.items()if key!='params'}
        selected['params']=[parameter]
        if 'param_names'in selected:selected['param_names']=[selected['param_names'][index]]
        options={'lr':group['lr'],**({'ns':child.ns}if type(child)is MuonSR else {})}
        shadow=type(child)([selected],**options)
        # Assign independent exact dtype-preserving state directly. Torch's
        # generic loader may otherwise cast historical moments to weight dtype.
        shadow.state[parameter]=_copy(child.state.get(p,{}))
        return parameter,shadow

    def abort(self):
        if self.buffer is not None:self.buffer.clear()
        self.buffer=None;self.aborted=True

    def begin_vjp_observation(self):
        if self.comparisons or self.buffer is not None:raise ValueError('optimizer observation already started')
        self.buffer={}

    def __call__(self,name,flat,cached):
        if self.completed is not None:raise ValueError('complete optimizer observer is already finalized')
        if name=='input_embeddings':return
        if name not in self.slots or name in self.comparisons:
            raise ValueError('complete update observer repeated or substituted a parameter')
        if self.buffer is not None:
            if name in self.buffer:raise ValueError('complete optimizer observer repeated a buffered parameter')
            # Keep RNG-changing shadow updates entirely outside autograd. Exact
            # CPU derivative copies retain their original dtype, never quantize.
            pair=tuple(None if value is None else value.detach().to('cpu',copy=True)for value in(flat,cached))
            self.buffer[name]=pair
            return
        self.assert_preserved();self.order.append(name)
        if flat is None or cached is None:
            self.comparisons[name]={'passed':flat is None and cached is None,
                'unused':flat is None and cached is None,'reason':'both unused'if flat is None and cached is None else'missing derivative'}
            return
        p=self.named[name]
        if flat.shape!=p.shape or cached.shape!=p.shape or flat.dtype!=p.dtype or cached.dtype!=p.dtype:
            raise ValueError('complete update observer gradient differs from actual parameter geometry/dtype')
        with torch.random.fork_rng(devices=self.devices):
            try:
                self._restore_rng(self.private_rng)
                left,left_optimizer=self._shadow(name,flat)
                left_optimizer.step()
                advanced=self._capture_rng()
                self._restore_rng(self.private_rng)
                right,right_optimizer=self._shadow(name,cached)
                right_optimizer.step()
                right_advanced=self._capture_rng()
                equal_rng=torch.equal(advanced['cpu'],right_advanced['cpu'])and all(
                    torch.equal(advanced['cuda'][device],right_advanced['cuda'][device])for device in self.devices)
                if not equal_rng:raise ValueError('counterfactual optimizer updates consumed different RNG shapes')
                self.private_rng=advanced
                gradients=gradient_metrics(cached,flat,atol=self.gradient_atol,rtol=self.gradient_rtol)
                before=p.detach().float();left_delta=left.detach().float()-before;right_delta=right.detach().float()-before
                delta=gradient_metrics(right_delta,left_delta,atol=self.update_atol,rtol=self.update_rtol)
                moments={}
                left_state,right_state=left_optimizer.state[left],right_optimizer.state[right]
                if set(left_state)!=set(right_state):raise ValueError('counterfactual optimizer state keys differ')
                for key,value in left_state.items():
                    other=right_state[key]
                    moments[key]=(gradient_metrics(other,value,atol=self.gradient_atol,rtol=self.gradient_rtol)
                        if torch.is_tensor(value)and value.is_floating_point()else
                        {'passed':bool(torch.equal(value,other))if torch.is_tensor(value)else type(value)is type(other)and value==other})
                count=_bytes(left_state)+_bytes(right_state)+_bytes(left)+_bytes(right)+_bytes(left.grad)+_bytes(right.grad)
                self.largest_pair_bytes=max(self.largest_pair_bytes,count)
                changed_left=int(torch.count_nonzero(left.detach()!=p.detach()))
                changed_right=int(torch.count_nonzero(right.detach()!=p.detach()))
                self.comparisons[name]={'passed':gradients['passed']and delta['passed']and all(row['passed']for row in moments.values()),
                    'optimizer':self.slots[name][0],'shape':list(p.shape),'dtype':str(p.dtype),
                    'gradient':gradients,'stored_update':delta,'moments':moments,
                    'learning_rate':float(self.slots[name][2]['lr']),'weight_decay':float(self.slots[name][2].get('weight_decay',0.)),
                    'optimizer_options':{'class':type(self.slots[name][1]).__name__,
                        'ns':getattr(self.slots[name][1],'ns',None),
                        'group':_metadata({key:value for key,value in self.slots[name][2].items()if key not in('params','param_names')})},
                    'flat_changed_coordinates':changed_left,'cached_changed_coordinates':changed_right,
                    'stored_weight_disagreement_coordinates':int(torch.count_nonzero(left.detach()!=right.detach())),
                    'flat_nonzero_gradient_coordinates':int(torch.count_nonzero(flat)),
                    'cached_nonzero_gradient_coordinates':int(torch.count_nonzero(cached)),
                    'zero_updates_with_nonzero_gradient':bool((not changed_left or not changed_right)and(torch.count_nonzero(flat)or torch.count_nonzero(cached)))}
            finally:self.assert_preserved()

    def finish(self):
        if self.aborted:raise ValueError('complete optimizer observation was aborted')
        if self.completed is not None:return self.completed
        if self.buffer is not None:
            waiting=self.buffer;self.buffer=None
            self.buffer_bytes=sum(_bytes(pair)for pair in waiting.values())
            try:
                for name in list(waiting):
                    pair=waiting.pop(name);device=self.named[name].device
                    self(name,*(None if value is None else value.to(device)for value in pair))
            finally:waiting.clear()
        self.assert_preserved()
        if set(self.comparisons)!=set(self.named):raise ValueError('complete optimizer comparison omitted a trainable parameter')
        self.completed={'schema':SCHEMA,'diagnostic_only':True,'readiness_qualification':False,
            'passed':all(row['passed']for row in self.comparisons.values()),
            'comparisons':self.comparisons,'parameter_order':self.order,
            'rng_coupling':'one private stream advanced once per parameter in recorded comparison order; both derivatives receive identical draws',
            'production_rng_order_equivalence_claimed':False,
            'optimizer_execution':{'muon_fused':os.environ.get('NATLANG_MUON_FUSED','1'),
                'muon_sr_noise':os.environ.get('NATLANG_MUON_SR_NOISE','philox')},
            'source_optimizer_updates':0,'source_tensors_preserved':True,
            'preservation_scope':'full VJP observation; later independent qualifier phases are outside this observer',
            'largest_parameter_state_pair_bytes':self.largest_pair_bytes,
            'retained_vjp_cpu_snapshot_bytes':self.buffer_bytes,
            'memory_scope':'CPU VJP snapshots plus shadow parameter/gradient/state pair; graph activations and autograd-returned gradients additional',
            'update_atol':self.update_atol,'update_rtol':self.update_rtol}

        return self.completed

    def report(self):
        return self.finish()
