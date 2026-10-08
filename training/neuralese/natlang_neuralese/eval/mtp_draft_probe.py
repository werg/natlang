"""Multi-token drafting of Neuralese block inputs vs the sketch rollout (frozen backbone).

Owner idea (2026-10-08): instead of Jacobi sketch passes (the shallow stack rerun over the whole sequence), draft
the next K block inputs chunk-wise from the full model's top state with one small shared module applied K-1 times
(DeepSeek-V3 MTP / EAGLE style, in embedding space). From the top state h_t, offset 1 is the full model's own greedy
input; the shared block then fuses (state, last drafted input) into the next state, whose readout (the sketch head
architecture: MLP correction, frozen final norm and LM head, argmax token embedding) drafts the following input.
Training is self-fed (the inference regime) with some teacher forcing; the chain at every position attends causally
to the same depth's chain at earlier positions, exactly as at inference.

Evaluation at sampled chunk starts c (gold context through c, K drafted inputs after it), for the drafts of
ar_greedy (the crisp autoregressive ceiling), ar_sketch (the trained sketch head's own rollout, which K sequence
passes reproduce) and mtp: per-offset draft token accuracy, and the full-stack consumer's gold CE with the drafts as
history, minus its CE with gold history (the same consumer as the autoregressive controls).

    python -m natlang_neuralese.eval.mtp_draft_probe --heads H --text-data T --out DIR [--offsets 8]
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..model.causal_feedback import CausalFeedbackProjection
from ..model.heads import RMSNorm
from ..serve import load_engine
from ..train.text_warmup import ROLE_CODES, chat_roles, chunked_readout, load_text_rows
from .projected_history import autoregressive_payloads


class SharedDrafter(nn.Module):
    """One dense causal block shared across draft depths, plus the sketch-architecture readout."""

    def __init__(self, backbone, depths, heads=16, ffn=4096):
        super().__init__()
        width = backbone.embedding_weight.shape[1]
        eps = backbone.norm_eps
        self.heads = heads
        self.state_norm, self.input_norm = RMSNorm(width, eps=eps), RMSNorm(width, eps=eps)
        self.fuse = nn.Linear(2 * width, width, bias=False)
        with torch.no_grad():  # start as the normalized state: well scaled for the frozen readout
            self.fuse.weight.zero_()
            self.fuse.weight[:, :width].copy_(torch.eye(width))
        self.depth = nn.Embedding(depths, width)
        nn.init.zeros_(self.depth.weight)
        self.attn_norm, self.ffn_norm = RMSNorm(width, eps=eps), RMSNorm(width, eps=eps)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.up = nn.Linear(width, 2 * ffn, bias=False)
        self.down = nn.Linear(ffn, width, bias=False)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.down.weight)
        self.readout = CausalFeedbackProjection(backbone)

    def step(self, state, drafted, depth):
        """[B,T,d] states and the inputs drafted for the next offset -> the next depth's states."""
        x = self.fuse(torch.cat((self.state_norm(state), self.input_norm(drafted)), -1)) + self.depth.weight[depth]
        b, t, d = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.out(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d))
        gate, up = self.up(self.ffn_norm(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * up)


def readout_argmax(head_logits, states, chunk=512):
    with torch.no_grad():
        return torch.cat([head_logits(s).argmax(-1) for s in states.split(chunk, 1)], 1)


def draft_chain(drafter, backbone, top, offsets, *, gold=None, teacher=False, loss_mask=None, sample=512):
    """Run the chain over every position. Returns drafted tokens [offsets, B, T] (offset j drafts the input of
    position t+1+j) and, with ``gold``, the summed CE of sampled positions per depth."""
    tokens = [readout_argmax(backbone.logits, top)]
    losses, state = [], top
    for depth in range(1, offsets):
        previous = gold[depth - 1] if teacher else tokens[-1]
        state = drafter.step(state, backbone.embed(previous).to(state.dtype), depth)
        tokens.append(readout_argmax(drafter.readout.logits, state))
        if gold is not None:
            valid = loss_mask[depth].nonzero(as_tuple=False)
            if len(valid):
                pick = valid[torch.randperm(len(valid), device=valid.device)[:sample]]
                logits = drafter.readout.logits(state[pick[:, 0], pick[:, 1]]).float()
                losses.append(F.cross_entropy(logits, gold[depth][pick[:, 0], pick[:, 1]]))
    return torch.stack(tokens), losses


def shifted_targets(ids, offsets, non_system):
    """gold[j][b,t] = ids[b,t+1+j] and mask[j] marks positions whose target exists and is not system."""
    b, t = ids.shape
    gold = torch.zeros(offsets, b, t, dtype=torch.long, device=ids.device)
    mask = torch.zeros(offsets, b, t, dtype=torch.bool, device=ids.device)
    for j in range(offsets):
        n = t - 1 - j
        if n > 0:
            gold[j, :, :n] = ids[:, 1 + j:]
            mask[j, :, :n] = non_system[:, 1 + j:]
    return gold, mask


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heads', type=Path, required=True)
    p.add_argument('--text-data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--offsets', type=int, default=8)
    p.add_argument('--segment', type=int, default=2048, help='non-system positions per window after the system prompt')
    p.add_argument('--max-context', type=int, default=6144, help='skip documents with a longer system prompt')
    p.add_argument('--train-windows', type=int, default=160)
    p.add_argument('--held-windows', type=int, default=12)
    p.add_argument('--starts', type=int, default=8, help='chunk starts per held window for the consumer comparison')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--teacher-fraction', type=float, default=0.25)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    if a.out.exists():
        raise ValueError('fresh output directory required')
    a.out.mkdir(parents=True)
    random.seed(a.seed); torch.manual_seed(a.seed)
    engine = load_engine(heads_checkpoint=str(a.heads), device=a.device)
    backbone, heads = engine.backbone, engine.heads
    backbone.eval()
    for q in list(backbone.parameters()) + list(heads.parameters()):
        q.requires_grad_(False)
    tokenizer = engine.tokenizer
    token_id = lambda text: tokenizer.convert_tokens_to_ids(text)
    role_ids = {token_id(n): n for n in ('system', 'user', 'assistant', 'tool')}
    system = ROLE_CODES.index('system')
    rows, receipt = load_text_rows(None, None, a.text_data, tokenizer=tokenizer)
    split = {'train': [], 'test': []}
    for row in rows:
        ids = [backbone.controls.open_id] + list(row['token_ids'] if 'token_ids' in row else engine._tokens(row['text']))
        roles = chat_roles(ids + [backbone.controls.close_id], start_id=token_id('<|im_start|>'), role_ids=role_ids,
                           think_open=token_id('<think>'), think_close=token_id('</think>'))[:len(ids)]
        context = 1
        while context < len(ids) and roles[context] in (0, system):
            context += 1
        if len(ids) - context > 4 * a.offsets and context <= a.max_context:
            split['train' if row['split'] == 'train' else 'test'].append(
                {'ids': ids[:context + a.segment], 'context': context,
                 'non_system': [r != system for r in roles[:context + a.segment]]})
    random.shuffle(split['train']); random.shuffle(split['test'])
    train, held = split['train'][:a.train_windows], split['test'][:a.held_windows]

    @torch.no_grad()
    def top_states(doc):
        """Full-model top states for the window's non-system part (the prefix is full context)."""
        ids = torch.tensor([doc['ids']], device=a.device)
        h = backbone.forward_embeds(backbone.embed(ids), logits=False)['h_final']
        lo = doc['context'] - 1  # the last system position drafts the first non-system input
        return h[:, lo:].to(torch.bfloat16).cpu(), ids[:, lo:].cpu(), torch.tensor([doc['non_system'][lo:]])

    started = time.perf_counter()
    train_data = [top_states(d) for d in train]
    print(json.dumps({'event': 'collected', 'train': len(train_data), 'held': len(held),
                      'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    drafter = SharedDrafter(backbone, a.offsets).to(a.device)
    trainable = [q for q in drafter.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s + 1) / 100) * max(0.05, 1 - s / a.steps))
    log = open(a.out / 'train.jsonl', 'w')
    for step in range(a.steps):
        top, ids, non_system = (x.to(a.device) for x in random.choice(train_data))
        gold, mask = shifted_targets(ids, a.offsets, non_system)
        teacher = random.random() < a.teacher_fraction
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, losses = draft_chain(drafter, backbone, top, a.offsets, gold=gold, teacher=teacher, loss_mask=mask)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step(); schedule.step()
        if step % 100 == 0 or step == a.steps - 1:
            row = {'step': step, 'teacher': teacher, 'loss': float(loss), 'per_depth': [round(float(x), 3) for x in losses]}
            log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
    drafter.eval()

    # Per-offset self-fed draft accuracy over all held non-system positions.
    accuracy = torch.zeros(a.offsets); counted = torch.zeros(a.offsets)
    consumer = {kind: {'ce': torch.zeros(a.offsets), 'draft_accuracy': torch.zeros(a.offsets),
                       'n': torch.zeros(a.offsets)} for kind in ('gold', 'ar_greedy', 'ar_sketch', 'mtp')}
    starts_used = 0
    for doc in held:
        top, ids, non_system = top_states(doc)
        top, ids, non_system = top.to(a.device), ids.to(a.device), non_system.to(a.device)
        gold, mask = shifted_targets(ids, a.offsets, non_system)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            drafts, _ = draft_chain(drafter, backbone, top, a.offsets)
        for j in range(a.offsets):
            accuracy[j] += float(((drafts[j] == gold[j]) & mask[j]).sum()); counted[j] += float(mask[j].sum())
        lo = doc['context'] - 1
        candidates = [t for t in range(ids.shape[1] - a.offsets - 1) if bool(mask[:, 0, t].all())]
        for t in sorted(random.sample(candidates, min(a.starts, len(candidates)))):
            c = lo + t  # chunk start in document coordinates: gold context ids[:c+1]
            prefix = torch.tensor([doc['ids'][:c + 1]], device=a.device)
            span = torch.tensor([doc['ids'][c + 1:c + 1 + a.offsets]], device=a.device)
            if span.shape[1] < a.offsets:
                continue
            with torch.no_grad():
                payloads, generated = autoregressive_payloads(backbone, heads, prefix, a.offsets,
                                                              ('ar_greedy', 'ar_sketch'), return_generated_tokens=True)
                kinds = {'gold': (backbone.embed(span), span),
                         'ar_greedy': (payloads['ar_greedy'], generated['ar_greedy']),
                         'ar_sketch': (payloads['ar_sketch'], generated.get('ar_sketch')),
                         'mtp': (backbone.embed(drafts[:, 0, t][None]), drafts[:, 0, t][None])}
                prefix_embeddings = backbone.embed(prefix)
                for kind, (payload, tokens) in kinds.items():
                    history = heads.read_embeddings(backbone, payload[:, :-1])
                    out = backbone.forward_embeds(torch.cat((prefix_embeddings, history), 1), logits=False)
                    states = out['h_final'][:, prefix.shape[1] - 1:]
                    _, _, _, _, losses = chunked_readout(backbone, states, span, backbone.controls.close_id,
                                                         gradients=False)
                    consumer[kind]['ce'] += losses[0].float().cpu()
                    if tokens is not None:
                        consumer[kind]['draft_accuracy'] += (tokens[0] == span[0]).float().cpu()
                    consumer[kind]['n'] += 1
            starts_used += 1
    gold_ce = consumer['gold']['ce'] / consumer['gold']['n'].clamp(min=1)
    results = {kind: {'ce_by_offset': (v['ce'] / v['n'].clamp(min=1)).tolist(),
                      'ce_delta_by_offset': (v['ce'] / v['n'].clamp(min=1) - gold_ce).tolist(),
                      'draft_accuracy_by_offset': (v['draft_accuracy'] / v['n'].clamp(min=1)).tolist()}
               for kind, v in consumer.items()}
    report = {'schema': 'natlang.mtp-draft-probe/1', 'heads': str(a.heads), 'inputs': receipt,
              'options': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
              'drafter_parameters': sum(q.numel() for q in trainable),
              'self_fed_draft_accuracy_by_offset': (accuracy / counted.clamp(min=1)).tolist(),
              'chunk_starts': starts_used, 'consumer': results,
              'offset_convention': 'offset j (0-based) drafts the input of position c+1+j; consumer CE at offset j '
                                   'scores gold token c+1+j given gold context through c and drafts c+1..c+j',
              'scope': 'frozen backbone and sketch head; diagnostic, not qualification'}
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'event': 'done', 'self_fed_draft_accuracy_by_offset': report['self_fed_draft_accuracy_by_offset'],
                      'ce_delta_by_offset': {k: [round(x, 3) for x in v['ce_delta_by_offset']]
                                             for k, v in results.items()}}), flush=True)


if __name__ == '__main__':
    main()
