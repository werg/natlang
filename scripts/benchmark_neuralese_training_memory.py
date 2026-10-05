#!/usr/bin/env python3
"""Measure a real port-training step with synthetic long-context stress inputs.

This is a resource check, not a quality evaluation or training-data generator.
Run each configuration in a fresh process so OOMs and allocator caches stay isolated.
"""
import argparse
import dataclasses
import json
import time
import traceback
from pathlib import Path

import torch
from natlang_neuralese.data.records import read_records
from natlang_neuralese.data.render import Renderer, render_record
from natlang_neuralese.model.heads import PortHeads
from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone, load_backbone
from natlang_neuralese.train.adapters import deltas_off, inject_lora
from natlang_neuralese.train.losses import consumer_batch_loss
from natlang_neuralese.train.optim import make_port_optimizer
from natlang_neuralese.train.memory import offload_attention_tensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student', required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--context', type=int, required=True)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--vectors', type=int, default=64)
    p.add_argument('--phase', choices=['D', 'E', 'F'], default='D')
    p.add_argument('--memory-gb', type=float, default=7)
    p.add_argument('--attention-offload-gb', type=float, default=0)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(a.memory_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    receipt = dict(context=a.context, batch=a.batch, vectors=a.vectors, phase=a.phase,
                   synthetic_stress_only=True, training_publication=False)
    started = time.time()
    try:
        model, tok = load_backbone(lora=a.student, device='cuda')
        backbone = PortBackbone(model, ControlTokens.from_tokenizer(tok))
        heads = PortHeads(backbone, cutoff=6, max_length=a.vectors, stop_source='final').to('cuda')
        optimizer = make_port_optimizer('muon', backbone, heads, lr=3e-4)
        if a.phase == 'F':
            for params in inject_lora(backbone, [15, 14, 13, 12], rank=16, alpha=32).values():
                optimizer.add_param_group({'params': params, 'lr': 1e-4})
        renderer = Renderer(tok, backbone.controls)
        examples = []
        for r in read_records([a.records], imitation_only=False):
            if r.split == 'train' and r.outcome_label in {'gold', 'checked', 'teacher'}:
                x = render_record(renderer, r, form='chat')
                if len(x.target) <= 192 and len(x.producer) < 1024:
                    examples.append(x)
                    if len(examples) == a.batch:
                        break
        assert len(examples) == a.batch
        # Ragged prefixes exercise the actual padding path. Targets and block markers
        # remain unchanged; filler is only for geometry and carries no quality claim.
        filler = renderer.text('Background context for a memory measurement. ')
        def padded(ids, length):
            n = length - len(ids)
            assert n >= 0
            return (filler * ((n + len(filler) - 1) // len(filler)))[:n] + ids
        batch = []
        for i, x in enumerate(examples):
            budget = a.context - a.vectors - len(x.target) - 16 - i * 257
            batch.append(dataclasses.replace(x,
                producer=padded(x.producer, budget),
                consumer_before=padded(x.consumer_before, budget - len(x.consumer_after)),
                teacher_prefix=padded(x.teacher_prefix, budget)))
        receipt['backbone_parameter_bytes'] = sum(q.numel() * q.element_size() for q in model.parameters())
        receipt['trainable_parameter_bytes'] = sum(q.numel() * q.element_size() for g in optimizer.param_groups for q in g['params'])
        receipt['optimizer_partition_parameters'] = {
            kind: sum(q.numel() for g in child.param_groups for q in g['params'])
            for kind, child in [('muon', optimizer.muon), ('adamw', optimizer.auxiliary)] if child}
        receipt['token_shapes'] = [dict(producer=len(x.producer), consumer_prefix=len(x.consumer_before), target=len(x.target)) for x in batch]
        # Force the maximum write length in this resource check, including phase E.
        # Its normal exploration often stops earlier and would understate worst-case RAM.
        with torch.no_grad():
            heads.stop.mlp_out.weight.zero_()
            heads.stop.mlp_out.bias.fill_(-100)
        with offload_attention_tensors(int(a.attention_offload_gb * 2**30)) as offload_stats:
            loss, metrics = consumer_batch_loss(backbone, heads, batch,
            max_length=a.vectors, temperature=.3, kl_weight=1,
            contrastive_weight=.5, diversity_weight=1,
            teacher_context=lambda: deltas_off(backbone),
            generator=torch.Generator().manual_seed(0),
            target_lengths=None if a.phase == 'E' else [a.vectors] * a.batch,
            policy_samples=2 if a.phase == 'E' else 1,
                stop_policy_weight=1 if a.phase == 'E' else 0)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(q.grad is None or torch.isfinite(q.grad).all() for g in optimizer.param_groups for q in g['params'])
        optimizer.step()
        torch.cuda.synchronize()
        receipt.update(status='passed', loss=float(loss.detach()), metrics=metrics,
                       attention_offload=offload_stats,
                       note='Includes forward, backward and first Muon/AdamW state allocation. Phase-F ordinary replay not included.')
    except torch.OutOfMemoryError as error:
        receipt.update(status='out_of_memory', error=str(error), traceback=traceback.format_exc())
    finally:
        receipt.update(seconds=time.time()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        a.output.write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
