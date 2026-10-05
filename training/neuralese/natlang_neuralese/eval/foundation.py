"""Token-aligned identity gate and causal-feedback diagnostics, before compression.

This gate never substitutes text for a learned payload. It tests the exact raw
embedding reference and a fresh zero-residual content projection separately from
the loaded learned channel. Passing the reference does not qualify that channel.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ..model.heads import ContentProjection
from ..serve import load_engine
from ..train.trajectories import target_write, handover_notes


def reference_next_embedding(backbone, final_states):
    """Exact greedy token reference: h[i] -> E(argmax LM(h[i])).

    For sampled generation, use that same sampled token instead of argmax.
    A probability-weighted embedding mixture is not this reference.
    """
    return backbone.embed(backbone.logits(final_states).argmax(-1))


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=512)
    args = parser.parse_args(argv)
    if args.limit < 1 or args.max_tokens < 2:
        parser.error('positive limit and at least two tokens required')
    if args.out.exists():
        parser.error('fresh immutable output required')
    texts = set()
    for row in map(json.loads, args.records.open()):
        name = target_write(row)
        if name and row.get('split') == 'test' and row.get('training_admission', {}).get('approved') is True:
            texts.add(handover_notes(row)[name])
    texts = sorted(texts, key=lambda text: hashlib.sha256(text.encode()).hexdigest())[:args.limit]
    if not texts:
        raise ValueError('no admitted held sources')
    torch.set_num_threads(2)
    engine = load_engine(heads_checkpoint=str(args.heads), device=args.device)
    backbone, heads = engine.backbone, engine.heads
    backbone.ffn_chunk_tokens = 2048
    fresh = ContentProjection(backbone.embedding_weight.shape[1]).to(args.device).eval()
    rows = []
    with torch.no_grad():
        for text in texts:
            ids = engine.tokenizer(text, add_special_tokens=False)['input_ids'][:args.max_tokens]
            if engine.tokenizer.bos_token_id is not None:
                ids = [engine.tokenizer.bos_token_id] + ids
            ids = torch.tensor([ids], device=args.device)
            raw = backbone.embed(ids)
            plain = backbone.forward_ids(ids, cutoff=heads.cutoff)
            # Slice already-tokenized positions; do not retokenize text fragments.
            start, end = 1, max(2, ids.shape[1] - 1)
            payload = fresh(raw[:, start:end], plain['h_final'][:, start:end])
            transport = torch.cat([raw[:, :start], payload, raw[:, end:]], dim=1)
            readback = backbone.forward_embeds(transport)
            logits = plain['logits']
            chosen = logits.argmax(-1)
            reference = reference_next_embedding(backbone, plain['h_final'])
            shallow_logits = heads.feedback.readout_logits(plain['h_cut'])
            feedback = heads.feedback(plain['h_cut']).float()
            target = backbone.embed(chosen).float()
            teacher = F.log_softmax(logits.float(), -1)
            student = F.log_softmax(shallow_logits.float(), -1)
            rows.append({
                'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'positions': ids.shape[1],
                'raw_transport_max_abs': float((transport - raw).abs().max()),
                'identity_readback_logits_max_abs': float((readback['logits'] - logits).abs().max()),
                'identity_readback_argmax_agreement': float((readback['logits'].argmax(-1) == chosen).float().mean()),
                'full_output_reference_max_abs': float((reference - backbone.embed(chosen)).abs().max()),
                'legacy_read_norm_relative_mse': float((heads.interface(raw).float() - raw.float()).square().mean() / raw.float().square().mean()),
                'learned_feedback_teacher_argmax_agreement': float((shallow_logits.argmax(-1) == chosen).float().mean()),
                'learned_feedback_teacher_kl': float((teacher.exp() * (teacher - student)).sum(-1).mean()),
                'learned_feedback_raw_greedy_embedding_relative_mse': float((feedback - target).square().mean() / target.square().mean()),
            })
            print(json.dumps(rows[-1]), flush=True)
    passed = all(row['raw_transport_max_abs'] == 0 and row['identity_readback_logits_max_abs'] == 0
                 and row['full_output_reference_max_abs'] == 0 for row in rows)
    report = {'schema': 'natlang.neuralese-foundation-control/1',
              'pins': {str(path): fingerprint(path) for path in (args.heads, args.records)},
              'options': vars(args) | {'heads': str(args.heads), 'records': str(args.records), 'out': str(args.out)},
              'token_aligned_reference_passed': passed, 'learned_channel_qualified': False,
              'scope': 'held source controls; raw identity is required but does not qualify learned feedback, compression or production transport',
              'rows': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise SystemExit('token-aligned identity reference failed')


if __name__ == '__main__':
    main()
