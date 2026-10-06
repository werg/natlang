"""Distill causal shallow-state feedback after checking exact full-depth identity.

This isolated stage freezes the entire backbone, retains raw embedding scale and
does not compress. A checkpoint is not a serving port until runtime integration
and task gates pass. Train and held sources have disjoint content identities.
"""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path

import torch
from torch.nn import functional as F

from ..model.causal_feedback import CausalFeedbackProjection, load_projection_state
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
    parser.add_argument('--continue-from', type=Path, help='explicit new-stage handoff preserving optimizer and RNG; not a changed-in-place resume')
    parser.add_argument('--cutoff', type=lambda value: 'full' if value == 'full' else int(value), default='full')
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
    parser.add_argument('--source-fraction', type=float, default=0.25)
    parser.add_argument('--argmax-weight', type=float, default=0.1)
    parser.add_argument('--stop-on-gate', action=argparse.BooleanOptionalAction, default=False,
                        help='qualify an already-correct initialization without unnecessary optimizer updates')
    args = parser.parse_args(argv)
    if min(args.steps, args.batch, args.tokens, args.eval_every, args.checkpoint_every) < 1:
        parser.error('positive bounds required')
    if (args.lr <= 0 or not 0 <= args.agreement_gate <= 1 or args.kl_gate < 0
            or not 0 <= args.source_fraction <= 1 or args.argmax_weight < 0):
        parser.error('invalid optimizer or gate controls')
    if args.contexts < 0 or args.context_tokens < 1 or (args.contexts and not args.pieces):
        parser.error('context curriculum requires pieces and positive bounds')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    package = Path(__file__).parents[1]
    code = sorted(package.rglob('*.py'))
    inputs = [args.heads, args.records] + ([args.pieces] if args.pieces else []) + ([args.continue_from] if args.continue_from else [])
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
    # Feed-forward rows per chunk: smaller chunks lower the peak where expert weights are rebuilt (Maple's ternary
    # experts on unified memory peaked near 50 GB at 2048).
    backbone.ffn_chunk_tokens = int(os.environ.get('NATLANG_FFN_CHUNK_TOKENS', 2048))
    cutoff = backbone.num_layers if args.cutoff == 'full' else args.cutoff
    if not 0 < cutoff <= backbone.num_layers:
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
    if args.stop_on_gate and cutoff == backbone.num_layers and not args.continue_from and not resumed:
        # The full-depth reference needs qualification, not optimization away
        # from an already exact solution. Stream held examples; do not cache
        # training features or simultaneous full-vocabulary sequence tensors.
        strata = {}
        with torch.no_grad():
            examples = {'source': [engine._tokens(text)[:args.tokens] for text in texts['test']],
                        'context': [list(ids) for ids in contexts['test']]}
            for name, sequences in examples.items():
                count = 0
                for ids in sequences:
                    if engine.tokenizer.bos_token_id is not None:
                        ids = [engine.tokenizer.bos_token_id] + ids
                    final = backbone.forward_ids(torch.tensor([ids], device=args.device), logits=False)['h_final']
                    for begin in range(0, final.shape[1], 64):
                        h = final[:, begin:begin + 64]
                        teacher, initialized = backbone.logits(h), projection.logits(h)
                        if not torch.equal(teacher, initialized):
                            raise AssertionError('full-depth copied readout is not exact')
                        if not torch.equal(projection(h), backbone.embed(teacher.argmax(-1))):
                            raise AssertionError('raw next-token embedding reference is not exact')
                        count += h.shape[1]
                if count:
                    strata[name] = {'kl': 0., 'agreement': 1., 'positions': count}
        if not strata:
            raise ValueError('no held reference examples')
        result = {'step': 0, 'kl': 0., 'agreement': 1., 'positions': sum(value['positions'] for value in strata.values()),
                  'strata': strata, 'feedback_gate_passed': True, 'full_output_bit_exact': True, 'runtime_qualified': False}
        named = [(name, value) for name, value in projection.named_parameters() if value.requires_grad]
        optimizer = PortMuonAdamW(named, lr=args.lr, vocab_size=backbone.embedding_weight.shape[0])
        state = {'schema': 'natlang.causal-feedback-bootstrap/1', 'identity': identity, 'step': 0,
                 'projection_state_format': 2,
                 'projection': projection.state_dict(), 'optimizer': optimizer.state_dict(),
                 'generator': torch.Generator().manual_seed(args.seed + 1).get_state(),
                 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'best': result,
                 'cutoff': cutoff, 'selection': 'greedy-raw-next-token',
                 'qualification': 'exact full-output initialization; runtime gate still required'}
        atomic_checkpoint(state_path, state)
        atomic_checkpoint(args.out / 'best-checkpoint.pt', state)
        (args.out / 'eval.jsonl').write_text(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
        print(json.dumps({'status': 'qualified_at_initialization', 'step': 0, 'optimizer_updates': 0,
                          'reason': 'exact full-output projection already satisfies the raw embedding distillation target'}), flush=True)
        return
    pairs, boundaries = {}, {}
    with torch.no_grad():
        for split in ['train', 'test']:
            values = []
            examples = [engine._tokens(text)[:args.tokens] for text in texts[split]] + [list(ids) for ids in contexts[split]]
            source_count = 0
            for index, ids in enumerate(examples):
                if not ids:
                    continue
                if engine.tokenizer.bos_token_id is not None:
                    ids = [engine.tokenizer.bos_token_id] + ids
                ids = torch.tensor([ids], device=args.device)
                out = backbone.forward_ids(ids, cutoff=cutoff, logits=False)
                reference = backbone.logits(out['h_final'])
                initialized = projection.logits(out['h_final'])
                if not torch.equal(reference, initialized):
                    raise AssertionError('copied full-output readout is not exactly equivalent')
                if not torch.equal(projection(out['h_final']), backbone.embed(reference.argmax(-1))):
                    raise AssertionError('full-depth next-token embedding reference failed')
                # Last source position is useful: it predicts the continuation too.
                values.append((out['h_cut'][0].detach(), out['h_final'][0].detach()))
                if index < len(texts[split]):
                    source_count += ids.shape[1]
            if not values:
                raise ValueError('empty ' + split + ' sources')
            pairs[split] = tuple(torch.cat([value[index] for value in values]) for index in [0, 1])
            boundaries[split] = source_count
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            print(json.dumps({'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30 if args.device == 'cuda' else 0,
                              'prepared': split, 'sources': len(values), 'positions': len(pairs[split][0]),
                              'context_windows': len(contexts[split]), 'full_depth_exact_identity': True}), flush=True)
    del values, out, reference, initialized
    # Cached-feature distillation only needs the final norm and vocabulary head.
    # Release obsolete preparation tensors and park unused transformer layers and
    # legacy heads on CPU rather than reserving GPU memory throughout warm-up.
    backbone.hf.model.layers.to('cpu')
    engine.heads.to('cpu')
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    print(json.dumps({'cached_feature_mode': True, 'transformer_layers_and_legacy_heads': 'cpu',
                      'gpu_allocated_gib': torch.cuda.memory_allocated() / 2**30}), flush=True)
    generator = torch.Generator().manual_seed(args.seed + 1)
    named = [(name, value) for name, value in projection.named_parameters() if value.requires_grad]
    optimizer = PortMuonAdamW(named, lr=args.lr, vocab_size=backbone.embedding_weight.shape[0])
    start, best = 0, None
    inherited = resumed
    if not resumed and args.continue_from:
        inherited = torch.load(args.continue_from, map_location='cpu', mmap=True, weights_only=False)
        if inherited.get('schema') != 'natlang.causal-feedback-bootstrap/1':
            raise ValueError('unrecognized bootstrap handoff')
        previous = inherited['identity']
        for path in [args.heads, args.records] + ([args.pieces] if args.pieces else []):
            if previous['inputs'].get(str(path)) != sha(path):
                raise ValueError('handoff teacher/data identity differs')
        for key in ['cutoff', 'lr', 'batch', 'tokens', 'contexts', 'context_tokens', 'seed']:
            if previous['options'].get(key) != options[key]:
                raise ValueError('handoff architecture/optimizer/data curriculum differs: ' + key)
    if inherited:
        load_projection_state(projection, inherited['projection'])
        optimizer.load_state_dict(inherited['optimizer'])
        generator.set_state(inherited['generator'])
        torch.set_rng_state(inherited['cpu_rng'])
        torch.cuda.set_rng_state_all(inherited['cuda_rng'])
        start = inherited['step']
        best = inherited['best'] if resumed else None
        if not resumed:
            print(json.dumps({'handoff_step': start, 'optimizer_and_rng_restored': True,
                              'new_stage_objective': {'source_fraction': args.source_fraction, 'argmax_weight': args.argmax_weight}}), flush=True)
    stop = [False]
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))

    def evaluate(step):
        strata = {}
        with torch.no_grad():
            shallow, final = pairs['test']
            for name, begin, end in [('source', 0, boundaries['test']), ('context', boundaries['test'], len(shallow))]:
                if begin == end:
                    continue
                statistics, count = {'kl': 0., 'agreement': 0.}, 0
                for offset in range(begin, end, args.batch):
                    h, target = shallow[offset:min(offset + args.batch, end)], final[offset:min(offset + args.batch, end)]
                    measured = metrics(projection.logits(h), backbone.logits(target))
                    for key in statistics:
                        statistics[key] += measured[key] * len(h)
                    count += len(h)
                strata[name] = {key: value / count for key, value in statistics.items()} | {'positions': count}
        count = sum(value['positions'] for value in strata.values())
        result = {key: sum(value[key] * value['positions'] for value in strata.values()) / count for key in ['kl', 'agreement']}
        result.update(step=step, positions=count, strata=strata,
                      feedback_gate_passed=all(value['agreement'] >= args.agreement_gate and value['kl'] <= args.kl_gate for value in strata.values()),
                      runtime_qualified=False)
        with (args.out / 'eval.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
        return result

    def save(step, path):
        atomic_checkpoint(path, {'schema': 'natlang.causal-feedback-bootstrap/1', 'identity': identity,
                                'projection_state_format': 2,
                                'step': step, 'projection': projection.state_dict(), 'optimizer': optimizer.state_dict(),
                                'generator': generator.get_state(), 'cpu_rng': torch.get_rng_state(),
                                'cuda_rng': torch.cuda.get_rng_state_all(), 'best': best,
                                'cutoff': cutoff, 'selection': 'greedy-raw-next-token',
                                'qualification': 'isolated distillation; runtime and recurrence gates still required'})

    if not resumed:
        initial = evaluate(start)
        best = initial
        save(start, state_path)
        save(start, args.out / 'best-checkpoint.pt')
        if args.stop_on_gate and initial['feedback_gate_passed']:
            print(json.dumps({'status': 'qualified_at_initialization', 'step': start,
                              'reason': 'copied full output head already reproduces the raw next-token embedding reference'}), flush=True)
            return
    projection.train()
    shallow, final = pairs['train']
    last_step = start
    for step in range(start, args.steps):
        boundary = boundaries['train']
        source_batch = round(args.batch * args.source_fraction) if boundary < len(shallow) else args.batch
        indices = torch.cat([torch.randint(boundary, (source_batch,), generator=generator),
                             torch.randint(boundary, len(shallow), (args.batch - source_batch,), generator=generator)
                             if source_batch < args.batch else torch.empty(0, dtype=torch.long)]).to(args.device)
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
        loss = kl + args.argmax_weight * ce + .1 * state_loss
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
            if args.stop_on_gate and result['feedback_gate_passed']:
                save(last_step, state_path)
                print(json.dumps({'status': 'qualified', 'step': last_step, 'best': best}), flush=True)
                return
        if last_step % args.checkpoint_every == 0 or stop[0] or last_step == args.steps:
            save(last_step, state_path)
        if stop[0]:
            break
    print(json.dumps({'status': 'checkpointed_on_signal' if stop[0] else 'complete', 'step': last_step, 'best': best}), flush=True)


if __name__ == '__main__':
    main()
