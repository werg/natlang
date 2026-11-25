"""Recipe T2, context-distillation part: train the site compiler on real calls (``plans/NATLANG_COMPILER.md`` §9.4).

    python -m natlang_neuralese.compiler.run_t2 prepare --collections DIR [DIR ...] --out ROWS.jsonl
    python -m natlang_neuralese.compiler.run_t2 train --rows ROWS.jsonl --checkpoint CK --base HF_DIR --out RUN

``prepare`` (no model): the calls of collected rollouts (``collect/*/groups/*/rollouts.jsonl``) whose rendered request
carries a site header (``compiled_site.site_header``) and whose episode succeeded; tasks excluded from training by
``held_eval.training_exclusion_ids`` (held-v3/v4 and reservations, siblings by split and source group, over the
groups' own case rows) are left out. Each site keeps at most ``--per-site`` calls; sites are split by a hash of their
header into train sites and unseen dev sites, and every train site leaves its last call (by hash) for dev.

``train``: the frozen policy as the Python native engine (serving dtype); per site the header's text state written in
its real call (cached in RUN/store); one shared ``xs`` adapter basis; ``CompiledSiteHead`` (FP32) trained with AdamW on
the native graph CE of the verified reply under the compiled stack. Receipts in RUN: ``train.jsonl`` (one line per
step), ``ablation.jsonl`` (base / joint / kv-only / adapter-only / shuffled-slots / wrong-site on dev calls of seen and
unseen sites: the mandatory shuffled-memory check), ``receipt.json`` (code and hindsight-kb provenance, engine
identity, data digest, config, exclusion funnel) and a compiled release of one dev site (``release/``) for the engine
parity check. ``learner-state.pt`` is saved atomically every ``--state-every`` steps and a rerun resumes from it.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

RECEIPT_SCHEMA = 'natlang.compiler.t2-receipt/1'


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _jsonl(path):
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare(args):
    from ..rl.held_eval import training_exclusion_ids
    from .compiled_site import site_binding, site_header
    groups = sorted({group for pattern in args.collections for group in glob.glob(os.path.join(pattern, 'groups', 'g*'))})
    cases = {}
    for group in groups:
        path = Path(group) / 'cases.jsonl'
        if path.exists():
            for row in _jsonl(path):
                cases.setdefault(row['id'], row)
    excluded = training_exclusion_ids(list(cases.values()))
    funnel = {'groups': len(groups), 'cases': len(cases), 'excluded_cases': len(excluded), 'rollouts': 0,
              'held_excluded_rollouts': 0, 'unknown_case_rollouts': 0, 'failed_rollouts': 0, 'calls': 0,
              'no_header': 0, 'too_long': 0, 'kept_calls': 0}
    by_site = {}
    for group in groups:
        path = Path(group) / 'rollouts.jsonl'
        if not path.exists():
            continue
        for rollout in _jsonl(path):
            funnel['rollouts'] += 1
            task = rollout['task_id']
            if task not in cases:
                funnel['unknown_case_rollouts'] += 1   # no case row to check its held closure: fail closed
                continue
            if task in excluded:
                funnel['held_excluded_rollouts'] += 1
                continue
            if rollout.get('outcome') != 'task_success':
                funnel['failed_rollouts'] += 1
                continue
            for index, turn in enumerate(rollout.get('turns') or ()):
                funnel['calls'] += 1
                sampling = turn.get('sampling') or {}
                header = site_header(turn.get('messages') or [])
                if header is None or not sampling:
                    funnel['no_header'] += 1
                    continue
                tokens = sum(len(part.get('token_ids') or ()) for part in sampling.get('prompt') or ())
                if tokens > args.max_prompt_tokens:
                    funnel['too_long'] += 1
                    continue
                site = _sha(header)
                key = _sha(json.dumps([rollout['task_id'], rollout.get('sample'), rollout.get('seed'), index]))
                by_site.setdefault(site, []).append({
                    'site': site, 'binding': site_binding(header), 'header': header, 'key': key,
                    'task_id': task, 'family': rollout.get('family'), 'source_groups': rollout.get('source_groups') or [],
                    'policy': (rollout.get('policy') or {}).get('checkpoint_sha256'), 'messages': turn['messages'],
                    'tools': turn.get('tools') or [], 'target': turn['target'], 'prompt_tokens': tokens})
                funnel['kept_calls'] += 1
    rows, sites = [], {'train': 0, 'dev': 0, 'dropped_small': 0}
    for site, calls in sorted(by_site.items()):
        calls = sorted({call['key']: call for call in calls}.values(), key=lambda call: call['key'])[:args.per_site]
        if len(calls) < args.min_per_site:
            sites['dropped_small'] += 1
            continue
        unseen = int(site[:8], 16) % 100 < args.dev_site_percent
        sites['dev' if unseen else 'train'] += 1
        for position, call in enumerate(calls):
            split = 'dev-unseen' if unseen else ('dev-seen' if position == len(calls) - 1 else 'train')
            rows.append({**call, 'split': split})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(temporary, out)
    summary = {'schema': 'natlang.compiler.t2-rows/1', 'rows': len(rows), 'sites': sites, 'funnel': funnel,
               'splits': {split: sum(1 for row in rows if row['split'] == split) for split in ('train', 'dev-seen', 'dev-unseen')},
               'collections': list(args.collections), 'max_prompt_tokens': args.max_prompt_tokens}
    out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


def _atomic_save(value, path):
    temporary = Path(str(path) + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(temporary, path)


class Sites:
    """Each site's text-initialized header state and each call's graph row, built once and cached in RUN/sites.json
    (the states themselves live in RUN/store)."""

    def __init__(self, engine, run):
        self.engine, self.path = engine, Path(run) / 'sites.json'
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {'states': {}}
        self.graphs = {}

    def state(self, row):
        from .compiled_site import init_header_state
        ident = self.cache['states'].get(row['site'])
        if ident is None or self.engine.store.get_native_state(ident) is None:
            ident = init_header_state(self.engine, row['messages'], row['tools'], row['header'])
            self.cache['states'][row['site']] = ident
            _write_json(self.path, self.cache)
        return ident

    def block(self, row):
        return self.engine.store.get_native_state(self.state(row)).block

    def row(self, row):
        from .compiled_site import site_graph
        if row['key'] not in self.graphs:
            state = self.state(row)
            block = self.engine.store.get_native_state(state).block
            graph = site_graph(self.engine, row['messages'], row['tools'], row['target'], row['header'], state,
                               len(block.token_ids), graph_id='t2:' + row['key'], source_groups=row['source_groups'] or [row['task_id']])
            self.graphs[row['key']] = {'graph': graph, 'state_id': state}
        return self.graphs[row['key']]


def _lr(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1., (step - args.warmup) / max(1, args.steps - args.warmup))
    return args.lr * (.1 + .9 * .5 * (1 + __import__('math').cos(__import__('math').pi * progress)))


def ablation(engine, head, sites, rows, adapter_id, step, others):
    """Mean dev CE per arm. ``shuffled``: the slots of another dev site; ``wrong_site``: the whole compiled stack of
    another site (its state values and cores) on this call. A compiler whose real slots do not beat the shuffled ones
    is a defect (AGENTS.md: a span that ablation cannot tell from shuffled state)."""
    from .compiled_site import site_loss
    head.eval()
    arms = {'base': [], 'joint': [], 'kv_only': [], 'adapter_only': [], 'shuffled': []}
    per_row = []
    with torch.no_grad():
        encoded = {row['site']: head.encode(_values(sites.block(row)))[0] for row in others}
    for index, row in enumerate(rows):
        site = sites.row(row)
        block = sites.block(row)
        other = next((o for o in others[index + 1:] + others[:index + 1] if o['site'] != row['site']), None)
        losses = {
            'base': site_loss(engine, head, site, adapter_id=adapter_id, init_block=block, kv=False, adapter=False),
            'joint': site_loss(engine, head, site, adapter_id=adapter_id, init_block=block),
            'kv_only': site_loss(engine, head, site, adapter_id=adapter_id, init_block=block, adapter=False),
            'adapter_only': site_loss(engine, head, site, adapter_id=adapter_id, init_block=block, kv=False),
        }
        if other is not None:
            losses['shuffled'] = site_loss(engine, head, site, adapter_id=adapter_id, init_block=block,
                                           z_slots=encoded[other['site']])
        for arm, value in losses.items():
            arms[arm].append(value)
        per_row.append({'key': row['key'], 'site': row['site'], 'split': row['split'], **losses})
    head.train()
    mean = {arm: (sum(values) / len(values) if values else None) for arm, values in arms.items()}
    paired = [r for r in per_row if 'shuffled' in r]
    wins = sum(1 for r in paired if r['joint'] < r['shuffled'])
    return {'step': step, 'rows': len(rows), 'mean_ce': mean,
            'base_minus_joint': None if mean['joint'] is None else mean['base'] - mean['joint'],
            'shuffled_minus_joint': None if mean['shuffled'] is None else mean['shuffled'] - mean['joint'],
            'joint_beats_shuffled': f'{wins}/{len(paired)}', 'per_row': per_row}


def _values(block):
    from .compiled_site import init_block_values
    return init_block_values(block)


def train(args):
    from ..common.code_provenance import code_provenance
    from ..common.hashing import sha256_file_hex
    from ..model.native_adapters import target_parameters
    from ..serve import load_native_engine
    from ..serve.store import TensorStore
    from .compiled_site import (CompiledSiteConfig, CompiledSiteHead, compile_release, create_compiler_adapter,
                                site_loss, state_layout)
    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = list(_jsonl(args.rows))
    train_rows = [row for row in rows if row['split'] == 'train']
    dev_rows = [row for row in rows if row['split'] != 'train']
    rng = random.Random(args.seed)
    dev_rows = rng.sample(dev_rows, min(len(dev_rows), args.dev_rows))
    engine = load_native_engine(str(args.checkpoint), base=str(args.base), device=args.device,
                                dtype=getattr(torch, args.dtype), training_state=False)
    engine.store = TensorStore(run / 'store')
    sites = Sites(engine, run)
    state_path = run / 'learner-state.pt'
    resumed = state_path.exists()
    saved = torch.load(state_path, map_location='cpu', weights_only=False) if resumed else None
    if saved is not None:
        config = CompiledSiteConfig.from_dict(saved['config'])
        adapter_id = saved['adapter_id']
    else:
        targets = tuple(sorted(target_parameters(engine.shared_model)))
        config = CompiledSiteConfig(layout=tuple(state_layout(sites.block(train_rows[0]))), targets=targets, rank=args.rank)
        adapter_id = create_compiler_adapter(engine, config.targets, config.rank)
    if engine.store.get_native_tensor_artifact(adapter_id) is None:
        raise ValueError('the run store lost its compiler adapter basis; this run cannot resume')
    head = CompiledSiteHead(config).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(.9, .95))
    step, order, position = 0, list(range(len(train_rows))), 0
    if saved is not None:
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        step, order, position = saved['step'], saved['order'], saved['position']
        torch.set_rng_state(saved['rng']['torch'])
        random.setstate(saved['rng']['python'])
    else:
        rng.shuffle(order)
    receipt = {'schema': RECEIPT_SCHEMA, 'code': code_provenance(), 'hindsight_kb': _hkb_provenance(),
               'engine_identity': dict(engine.native_identity), 'checkpoint': str(args.checkpoint),
               'checkpoint_sha256': args.checkpoint_sha256, 'base': str(args.base), 'rows': str(args.rows),
               'rows_sha256': sha256_file_hex(args.rows), 'rows_summary': _summary(args.rows), 'config': config.to_dict(),
               'adapter_id': adapter_id, 'args': {key: str(value) for key, value in vars(args).items() if key != 'func'},
               'parameters': sum(p.numel() for p in head.parameters()),
               'parameter_dtypes': sorted({str(p.dtype) for p in head.parameters()}), 'resumed_at_step': step if resumed else None,
               'objective': 'native-graph-token-ce/1 on verified successful replies, compiled stack (xs cores + header state)'}
    _write_json(run / ('receipt.json' if not resumed else f'receipt-resume-{step}.json'), receipt)
    log = (run / 'train.jsonl').open('a', encoding='utf-8')
    abl = (run / 'ablation.jsonl').open('a', encoding='utf-8')
    if step == 0 and not resumed:
        result = ablation(engine, head, sites, dev_rows, adapter_id, 0, dev_rows)
        abl.write(json.dumps(result) + '\n'); abl.flush()
    head.train()
    while step < args.steps:
        started = time.time()
        for group in optimizer.param_groups:
            group['lr'] = _lr(step, args)
        optimizer.zero_grad(set_to_none=True)
        losses, tokens = [], 0
        for _ in range(args.batch):
            if position == len(order):
                rng.shuffle(order)
                position = 0
            row = train_rows[order[position]]
            position += 1
            site = sites.row(row)
            tokens += len(site['graph']['nodes'][0]['token_ids'])
            losses.append(site_loss(engine, head, site, adapter_id=adapter_id, init_block=sites.block(row),
                                    backward=True, scale=1. / args.batch))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), args.clip))
        optimizer.step()
        step += 1
        encoder_grad = sum(float(p.grad.norm()) ** 2 for p in head.encoder.parameters() if p.grad is not None) ** .5
        log.write(json.dumps({'step': step, 'loss': sum(losses) / len(losses), 'losses': losses, 'grad_norm': grad_norm,
                              'encoder_grad_norm': encoder_grad, 'lr': optimizer.param_groups[0]['lr'], 'tokens': tokens,
                              'seconds': round(time.time() - started, 3),
                              'cuda_peak_gib': (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else None}) + '\n')
        log.flush()
        if step % args.state_every == 0 or step == args.steps:
            _atomic_save({'head': head.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'order': order,
                          'position': position, 'config': config.to_dict(), 'adapter_id': adapter_id,
                          'rng': {'torch': torch.get_rng_state(), 'python': random.getstate()}}, state_path)
        if step % args.eval_every == 0 or step == args.steps:
            result = ablation(engine, head, sites, dev_rows, adapter_id, step, dev_rows)
            abl.write(json.dumps(result) + '\n'); abl.flush()
    release_row = next((row for row in dev_rows if row['split'] == 'dev-seen'), dev_rows[0])
    compile_release(engine, head, adapter_id=adapter_id, init_block=sites.block(release_row), out=run / 'release',
                    site_id=release_row['site'], provenance={'receipt': receipt['code'], 'step': step,
                                                             'row': release_row['key'], 'header': release_row['header']})
    (run / 'release' / 'site-row.json').write_text(json.dumps(release_row))
    return 0


def _summary(rows_path):
    path = Path(rows_path).with_suffix('.summary.json')
    return json.loads(path.read_text()) if path.exists() else None


def _hkb_provenance():
    try:
        from .hkb_dependency import provenance
        return provenance()
    except ModuleNotFoundError as error:
        return {'unavailable': str(error)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--collections', nargs='+', required=True, help='collection directories (each with groups/g*/)')
    p.add_argument('--out', required=True)
    p.add_argument('--max-prompt-tokens', type=int, default=4096)
    p.add_argument('--per-site', type=int, default=8)
    p.add_argument('--min-per-site', type=int, default=2)
    p.add_argument('--dev-site-percent', type=int, default=10)
    p.set_defaults(func=prepare)
    t = sub.add_parser('train')
    t.add_argument('--rows', type=Path, required=True)
    t.add_argument('--checkpoint', type=Path, required=True)
    t.add_argument('--checkpoint-sha256', default=None)
    t.add_argument('--base', type=Path, required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--device', default='cuda')
    t.add_argument('--dtype', default='bfloat16', help="the frozen policy's serving dtype")
    t.add_argument('--steps', type=int, default=200)
    t.add_argument('--batch', type=int, default=4)
    t.add_argument('--lr', type=float, default=2e-4)
    t.add_argument('--weight-decay', type=float, default=.01)
    t.add_argument('--warmup', type=int, default=20)
    t.add_argument('--clip', type=float, default=1.)
    t.add_argument('--rank', type=int, default=16)
    t.add_argument('--dev-rows', type=int, default=24)
    t.add_argument('--eval-every', type=int, default=50)
    t.add_argument('--state-every', type=int, default=25)
    t.add_argument('--seed', type=int, default=0)
    t.set_defaults(func=train)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
