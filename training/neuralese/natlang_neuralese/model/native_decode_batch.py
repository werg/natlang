"""Layer-major incremental decoding with shared projection/FFN work.

Each lane owns its positional KV and hybrid state. Only token-local model math
is combined across lanes; attention and recurrent replay keep exact lane-local
positions and state. Captures have the same schema as native_step. This is also
differentiable for comparison with independently cached executions.
"""
from __future__ import annotations
import torch
from . import shared_state as shared
from .lfm2_port import PortCache, ConvState, DeltaState, append_kv


def validate_decode_cache(port, cache):
    if cache is None:return
    if cache.pad is not None or len(cache.states)!=port.num_layers or len(cache.lengths)!=port.num_layers or len(set(cache.lengths))!=1:
        raise ValueError('native state requires an unpadded cache with aligned layer positions')


def native_step_batch(model, ids, caches, *, producing=None):
    """Execute one complete token per independent lane with common model weights."""
    if ids.ndim!=2 or ids.shape[1]!=1 or ids.shape[0]!=len(caches) or not len(caches):
        raise ValueError('native decode batch requires [lanes,1] tokens and one cache per lane')
    count=len(caches);port=model.port
    for cache in caches:validate_decode_cache(port,cache)
    caches=[cache or PortCache.empty(port.num_layers) for cache in caches]
    starts=[cache.lengths[0] for cache in caches]
    flags=[False]*count if producing is None else list(producing)
    if len(flags)!=count or any(type(flag)is not bool for flag in flags):
        raise ValueError('native decode batch requires one boolean producer role per lane')
    raw=model.embed(ids.transpose(0,1))
    mask=torch.tensor([flags],dtype=torch.bool,device=ids.device)
    h=raw if not any(flags) and not torch.is_grad_enabled() else model.adapter(raw,mask)
    positions=[torch.arange(start,start+1,device=ids.device) for start in starts]
    layouts=[shared.AttentionLayout((start,),start+1) for start in starts]
    states=[[]for _ in caches];captures=[[]for _ in caches];rotary={};masks={}
    for layer_index,layer in enumerate(port.layers):
        if shared._is_gated_delta(port,layer_index):
            from .native_gated_delta import project_inputs,replay
            projected=project_inputs(layer.linear_attn,layer.input_layernorm(h),lambda linear,value:linear(value))
            outputs=[]
            for lane,cache in enumerate(caches):
                payload={name:value[0,lane:lane+1]for name,value in zip(shared.PAYLOAD_FIELDS,projected)}
                previous=cache.states[layer_index]
                out,conv,recurrent=replay(layer.linear_attn,payload,
                    conv_prefix=previous.conv if isinstance(previous,DeltaState) else None,
                    recurrent_state=previous.recurrent if isinstance(previous,DeltaState) else None)
                outputs.append(out);states[lane].append(DeltaState(conv,recurrent))
                captures[lane].append({'kind':'gated_delta',**payload})
            h=h+torch.cat(outputs,dim=1)
            h=port._feed_forward(layer,h)
            continue
        attention=port.is_attention(layer_index)
        values=shared._layer_inputs(port,layer,h,attention)
        if attention:
            q,k,v,*gates=values;outputs=[]
            for lane,cache in enumerate(caches):
                q_lane=q[:,:,lane:lane+1];k_lane=k[:,:,lane:lane+1];v_lane=v[:,:,lane:lane+1]
                captures[lane].append({'kind':'attention','raw_k':k_lane[0].transpose(0,1),'v':v_lane[0].transpose(0,1)})
                key=(q.device,q.dtype,starts[lane])
                if key not in rotary:rotary[key]=shared._rotary_tables(port,q_lane,positions[lane])
                query,rotated=shared._rotate_pair(port,q_lane,k_lane.to(q.dtype),positions[lane],layer_index,tables=rotary[key])
                value=v_lane.to(q.dtype)
                if shared._fp32_kv_cotangents(port,q_lane):rotated,value=rotated.float(),value.float()
                state=append_kv(cache.states[layer_index],rotated,value,static=not torch.is_grad_enabled())
                window=port.window(layer_index)if hasattr(port,'window')else None
                key=(starts[lane],window)
                if key not in masks:
                    masks[key]=shared.graph_block_mask(layouts[lane],ids.device,window)if ids.is_cuda else layouts[lane].attention_mask(ids.device,window)
                out=shared.graph_attention(query,state.k,state.v,layouts[lane],window,layer.self_attn.scaling,
                    prepared_mask=masks[key],precision=getattr(port,'native_attention_precision','ieee'))
                if gates:
                    gate=gates[0][:,lane:lane+1]
                    flat=out.transpose(1,2).reshape(1,1,-1)*torch.sigmoid(gate)
                    out=flat.view(1,1,-1,layer.self_attn.head_dim).transpose(1,2)
                outputs.append(out);states[lane].append(state)
            out=torch.cat(outputs,dim=2)
        else:
            gated,c=values
            prefixes=[]
            for lane,cache in enumerate(caches):
                previous=cache.states[layer_index]
                prefixes.append(gated.new_zeros(1,gated.shape[-1],port.conv_window)if previous is None else previous.window)
                captures[lane].append({'kind':'finite_convolution','gated_inputs':gated[0,lane:lane+1]})
            # Batch each independent finite history, then return to token-local layout.
            full=torch.cat([torch.cat(prefixes,dim=0),gated.transpose(0,1).transpose(1,2)],dim=2)
            if full.dtype==torch.float64:
                from .state_gather import gather_convolution_state
                indices=torch.arange(layer.conv.conv.weight.shape[-1],device=ids.device)[None]
                histories=gather_convolution_state(full.transpose(1,2),indices)
            else:histories=full.unfold(2,layer.conv.conv.weight.shape[-1],1).permute(0,2,3,1)
            out=shared._finite_convolution(layer.conv.conv,histories,c.transpose(0,1)).transpose(0,1)
            for lane in range(count):states[lane].append(ConvState(full[lane:lane+1,:,-port.conv_window:]))
        h=shared._finish(port,layer,h,out,attention)
    return [{'h_final':h[:,lane:lane+1],
        'cache':PortCache(tuple(states[lane]),(starts[lane]+1,)*port.num_layers),
        'token_ids':ids[lane],'embeddings':raw[0,lane:lane+1],
        'positions':positions[lane],'layers':tuple(captures[lane])}for lane in range(count)]
