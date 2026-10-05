"""Distill causal shallow-state feedback after checking exact full-depth identity.

This isolated stage freezes the entire backbone, retains raw embedding scale and
does not compress. A checkpoint is not a serving port until runtime integration
and task gates pass. Train and held sources have disjoint content identities.
"""
import argparse
import hashlib
import json
import signal
from pathlib import Path

import torch
from torch.nn import functional as F

from ..model.causal_feedback import CausalFeedbackProjection
from ..serve import load_engine
from .optim import PortMuonAdamW
from .output_embedding_projection import sha, source_texts
from .trajectory_state import atomic_checkpoint
from .bootstrap_data import context_ids


def metrics(predicted, teacher):
    target = F.log_softmax(teacher.float(), -1)
    student = F.log_softmax(predicted.float(), -1)
    return {'kl': float((target.exp() * (target - student)).sum(-1).mean()),
            'agreement': float((predicted.argmax(-1) == teacher.argmax(-1)).float().mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--pieces', type=Path)
    parser.add_argument('--contexts', type=int, default=0)
    parser.add_argument('--context-tokens', type=int, default=1024)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cutoff', type=int, default=14)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--tokens', type=int, default=1024)
    parser.add_argument('--eval-every', type=int, default=128)
    parser.add_argument('--checkpoint-every', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--agreement-gate', type=float, default=0.9)
    parser.add_argument('--kl-gate', type=float, default=0.25)
    args = parser.parse_args(argv)
    if min(args.steps, args.batch, args.tokens, args.eval_every, args.checkpoint_every) < 1:
        parser.error('positive bounds required')
    if args.lr <= 0 or not 0 <= args.agreement_gate <= 1 or args.kl_gate < 0:
        parser.error('invalid optimizer or gate controls')
    if args.contexts < 0 or args.context_tokens < 1 or (args.contexts and not args.pieces):
        parser.error('context curriculum requires pieces and positive bounds')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    package = Path(__file__).parents[1]
    code = sorted(package.rglob('*.py'))
    inputs = [args.heads, args.records] + ([args.pieces] if args.pieces else [])
    identity = {'options': options, 'inputs': {str(path): sha(path) for path in inputs},
                'code': {str(path.relative_to(package)): sha(path) for path in code}}
    state_path = args.out / 'checkpoint.pt'
    resumed = torch.load(state_path, map_location='cpu', mmap=True, weights_only=False) if state_path.exists() else None
    if resumed and (resumed.get('schema') != 'natlang.causal-feedback-bootstrap/1' or resumed['identity'] != identity):
        raise ValueError('bootstrap resume identity differs')
    if args.out.exists() and not resumed:
        raise ValueError('output exists without complete resumable checkpoint')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'plan.json').write_text(json.dumps(identity, indent=2) + '\n')
    engine = load_engine(heads_checkpoint=str(args.heads), device=args.device)
    backbone = engine.backbone
    backbone.ffn_chunk_tokens = 2048
    if not 0 < args.cutoff <= backbone.num_layers:
        raise ValueError('cutoff outside the stack')
    projection = CausalFeedbackProjection(backbone).to(args.device).eval()
    texts = source_texts(args.records)
    shared = set(texts['train']) & set(texts['test'])
    # Shared library definitions are legitimate inputs in both fixture splits,
    # but they cannot count as independently held projection examples.
    texts['test'] = [text for text in texts['test'] if text not in shared]
    (args.out / 'split-review.json').write_text(json.dumps({
        'policy': 'exclude exact train-source duplicates from held projection scoring',
        'excluded_held_sha256': sorted(hashlib.sha256(text.encode()).hexdigest() for text in shared),
        'train_sources': len(texts['train']), 'held_sources': len(texts['test']),
    }, indent=2) + '\n')
    contexts = {'train': [], 'test': []}
    if args.contexts:
        contexts, review = context_ids(engine, args.records, args.pieces, args.contexts, args.context_tokens)
        (args.out / 'context-review.json').write_text(json.dumps(review, indent=2) + '\n')
    pairs = {}
    with torch.no_grad():
        for split in ['train', 'test']:
            values = []
            examples = [engine._tokens(text)[:args.tokens] for text in texts[split]] + [list(ids) for ids in contexts[split]]
            for ids in examples:
                if not ids:
                    continue
                if engine.tokenizer.bos_token_id is not None:
                    ids = [engine.tokenizer.bos_token_id] + ids
                ids = torch.tensor([ids], device=args.device)
                out = backbone.forward_ids(ids, cutoff=args.cutoff, logits=False)
                reference = backbone.logits(out['h_final'])
                initialized = projection.logits(out['h_final'])
                if not torch.equal(reference, initialized):
                    raise AssertionError('copied full-output readout is not exactly equivalent')
                if not torch.equal(projection(out['h_final']), backbone.embed(reference.argmax(-1))):
                    raise AssertionError('full-depth next-token embedding reference failed')
                # Last source position is useful: it predicts the continuation too.
                values.append((out['h_cut'][0].detach(), out['h_final'][0].detach()))
            if not values:
                raise ValueError('empty ' + split + ' sources')
            pairs[split] = tuple(torch.cat([value[index] for value in values]) for index in [0, 1])
            print(json.dumps({'prepared': split, 'sources': len(values), 'positions': len(pairs[split][0]),
                              'context_windows': len(contexts[split]), 'full_depth_exact_identity': True}), flush=True)
    generator = torch.Generator().manual_seed(args.seed + 1)
    named = [(name, value) for name, value in projection.named_parameters() if value.requires_grad]
    optimizer = PortMuonAdamW(named, lr=args.lr, vocab_size=backbone.embedding_weight.shape[0])
    start, best = 0, None
    if resumed:
        projection.load_state_dict(resumed['projection'])
        optimizer.load_state_dict(resumed['optimizer'])
        generator.set_state(resumed['generator'])
        torch.set_rng_state(resumed['cpu_rng'])
        torch.cuda.set_rng_state_all(resumed['cuda_rng'])
        start, best = resumed['step'], resumed['best']
    stop = [False]
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))

    def evaluate(step):
        statistics, count = {'kl': 0., 'agreement': 0.}, 0
        with torch.no_grad():
            shallow, final = pairs['test']
            for offset in range(0, len(shallow), args.batch):
                h, target = shallow[offset:offset + args.batch], final[offset:offset + args.batch]
                result = metrics(projection.logits(h), backbone.logits(target))
                for key in statistics:
                    statistics[key] += result[key] * len(h)
                count += len(h)
        result = {key: value / count for key, value in statistics.items()}
        result.update(step=step, positions=count,
                      feedback_gate_passed=result['agreement'] >= args.agreement_gate and result['kl'] <= args.kl_gate,
                      runtime_qualified=False)
        with (args.out / 'eval.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
        return result

    def save(step, path):
        atomic_checkpoint(path, {'schema': 'natlang.causal-feedback-bootstrap/1', 'identity': identity,
                                'step': step, 'projection': projection.state_dict(), 'optimizer': optimizer.state_dict(),
                                'generator': generator.get_state(), 'cpu_rng': torch.get_rng_state(),
                                'cuda_rng': torch.cuda.get_rng_state_all(), 'best': best,
                                'cutoff': args.cutoff, 'selection': 'greedy-raw-next-token',
                                'qualification': 'isolated distillation; runtime and recurrence gates still required'})

    if not resumed:
        evaluate(0)
        save(0, state_path)
    projection.train()
    shallow, final = pairs['train']
    last_step = start
    for step in range(start, args.steps):
        indices = torch.randint(len(shallow), (args.batch,), generator=generator).to(args.device)
        h, target = shallow[indices], final[indices]
        with torch.no_grad():
            teacher = backbone.logits(target).float()
            probabilities = teacher.softmax(-1)
        predicted = projection.logits(h).float()
        log_prob = predicted.log_softmax(-1)
        kl = (probabilities * (teacher.log_softmax(-1) - log_prob)).sum(-1).mean()
        ce = F.cross_entropy(predicted, teacher.argmax(-1))
        corrected = projection.complete_state(h).float()
        state_loss = F.mse_loss(F.normalize(corrected, dim=-1), F.normalize(target.float(), dim=-1)) * corrected.shape[-1]
        loss = kl + .5 * ce + .1 * state_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([value for _, value in named], 1.)
        optimizer.step()
        last_step = step + 1
        if last_step % 32 == 0:
            print(json.dumps({'step': last_step, 'loss': float(loss.detach()), 'kl': float(kl.detach()),
                              'gradient_norm': float(grad), 'peak_gib': torch.cuda.max_memory_allocated() / 2**30}), flush=True)
        if last_step % args.eval_every == 0:
            result = evaluate(last_step)
            if best is None or result['kl'] < best['kl']:
                best = result
                save(last_step, args.out / 'best-checkpoint.pt')
        if last_step % args.checkpoint_every == 0 or stop[0] or last_step == args.steps:
            save(last_step, state_path)
        if stop[0]:
            break
    print(json.dumps({'status': 'checkpointed_on_signal' if stop[0] else 'complete', 'step': last_step, 'best': best}), flush=True)


if __name__ == '__main__':
    main()
