"""Recipe T2: train the site compiler on real calls (``plans/NATLANG_COMPILER.md`` §9.4).

    python -m natlang_neuralese.compiler.run_t2 prepare --collections DIR [DIR ...] [--call-level DIR ...] --out ROWS.jsonl
    python -m natlang_neuralese.compiler.run_t2 train --rows ROWS.jsonl --checkpoint CK --base HF_DIR --out RUN

``prepare`` (no model): the calls of collected rollouts (``collect/*/groups/*/rollouts.jsonl``) whose rendered request
carries a site header (``compiled_site.site_header``) and whose episode succeeded, plus Engine A's verified call-level
teacher answers (``call_level_engine.py teach``: ``DIR/call-level.jsonl.zst``; the teacher's verified answer is the
target, its reasoning is not trained). Tasks excluded from training by ``held_eval.training_exclusion_ids`` (held-v3/v4
and reservations, siblings by split and source group, over the collections' own case rows) are left out; a call-level
row whose task has no case row in the given collections is left out too (fail closed). Calls importing an episode's
native state are left out and counted. Each site keeps at most ``--per-site`` calls; sites are split by a hash of their
header into train sites and unseen dev sites, and every train site leaves its last call (by hash) for dev.

``train``: the frozen policy as the Python native engine (serving dtype); per site the header's text state written in
its real call (cached in RUN/store); one shared ``xs`` adapter basis, learned jointly by default (``LearnedBankBasis``,
``--no-learn-basis`` keeps the weight-SVD basis); ``CompiledSiteHead`` (FP32) trained with AdamW (FP32 masters) on the native graph CE
of the verified reply under the compiled stack, with independent ``kv_drop`` / ``adapter_drop``. A teacher reply's
think block (empty or not) and lead-in, and an own reply's unsampled empty think block, are zero-weight context
(``serve.targets.target_context_end``; ``--no-mask-non-student-context`` opts out). Each row of a batch counts by its
target tokens up to ``--row-token-cap`` (``--loss-weighting rows``: equally). Alongside, by default:
- Live-KB: each row gets slots from related peers (same family or same site; never its own task or a source-group
  sibling), filled by similarity from the writer's key index of the train pool (``--kb-index-every``) where it has too
  few, and unrelated distractors encoded with gradient (``--no-kb-neg-grad``); ``--null-kb-prob`` of rows see
  distractors only (restraint). The writer is hindsight-kb's
  pplx ``H``/``Q`` warm-started from 000000002 (``--writer-backend pplx --warmstart-dir --pplx-revision``; fresh
  ``value_head`` and poolers at ``--lr``, warm encoders at ``--warm-lr``; FP32 masters via MasterAdamW); the embedding
  stand-in (``--writer-backend embed``) runs only on CPU, for unit tests.
- FO-MAML every ``--meta-every`` steps on a multi-call site: inner SGD on ``z_site`` over the support calls, the query
  call's loss at ``z0 + sg(z_K - z0)`` into the encoder, head and basis, plus a Reptile pull.
- Compositional cores every ``--comp-every`` steps: a call compiled as ``C_scope + C_body`` (signature line and
  instructions body compiled separately, summed in the shared basis) with the full header's state residual.
Receipts in RUN: ``train.jsonl`` (one line per step), ``ablation.jsonl`` (base / joint / kv-only / adapter-only /
site-blind mean core / whole shuffled site / shuffled site slots / shuffled KB slots on dev calls of seen and unseen
sites, overall, by source kind and token-weighted: the mandatory shuffled-memory check),
``receipt.json`` (code and hindsight-kb provenance, engine identity, data digest, config, exclusion funnel) and a
compiled release of one dev site (``release/``) for the engine parity check. ``learner-state.pt`` (head, basis, writer,
optimizer, order, RNG) is saved atomically every ``--state-every`` steps and a rerun resumes from it.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch

RECEIPT_SCHEMA = 'natlang.compiler.t2-receipt/2'
# /3 (2026-11-26): non-student target context masked, token-weighted rows, similarity-filled Live-KB positives with
# distractor gradients. A learner state saved under another objective is not resumed (a head fit to /2 learned one
# global adapter under the wrong objective; T2 restarts from the warm start).
OBJECTIVE = 'unified-compiler-live-kb-maml-compositional/4'


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _jsonl(path):
    p = Path(path)
    if str(p).endswith('.zst'):
        import io
        import zstandard as zstd
        with p.open('rb') as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                for line in io.TextIOWrapper(reader, encoding='utf-8'):
                    if line.strip():
                        yield json.loads(line)
    else:
        with p.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _experience_summary(header: str, target: dict) -> str:
    """Compact verified demonstration text encoded by the Live-KB writer."""
    sig = header.splitlines()[0] if header else ''
    content = ''
    if isinstance(target, dict):
        if target.get('content'):
            content = str(target['content'])[:240]
        elif target.get('tool_calls'):
            tc = target['tool_calls'][0]
            fn = tc.get('function', {}) if isinstance(tc, dict) else {}
            content = f"{fn.get('name', '')}({str(fn.get('arguments', ''))[:200]})"
    return f'{sig}\nVerified reply: {content}'.strip()


def imports_native_state(messages):
    """Whether a call's messages carry a native-state part (an episode's Neuralese value or compaction capture)."""
    def walk(value):
        if isinstance(value, dict):
            return value.get('type') == 'native_state' or any(walk(item) for item in value.values())
        if isinstance(value, list):
            return any(walk(item) for item in value)
        return False
    return walk(messages)


def prepare(args):
    from ..rl.held_eval import training_exclusion_ids
    from .compiled_site import site_binding, site_header, split_compositional_header

    groups = sorted({group for pattern in (args.collections or ()) for group in glob.glob(os.path.join(pattern, 'groups', 'g*'))})
    cases = {}
    for group in groups:
        path = Path(group) / 'cases.jsonl'
        if path.exists():
            for row in _jsonl(path):
                cases.setdefault(row['id'], row)
    for extra_cases in (getattr(args, 'cases', None) or ()):
        # case rows, or rows wrapping one (an authored pool's authored.jsonl: {id, task: CASE, ...}); a missing file
        # or a row without a case id fails: held closure is checked over these rows
        for row in _jsonl(extra_cases):
            case = row['task'] if isinstance(row.get('task'), dict) else row
            if not case.get('id'):
                raise ValueError(f'{extra_cases}: a row without a case id')
            cases.setdefault(case['id'], case)

    excluded = training_exclusion_ids(list(cases.values()))
    funnel = {
        'groups': len(groups), 'cases': len(cases), 'excluded_cases': len(excluded),
        'rollouts': 0, 'held_excluded_rollouts': 0, 'unknown_case_rollouts': 0, 'failed_rollouts': 0,
        'calls': 0, 'no_header': 0, 'imports_native_state': 0, 'too_long': 0, 'kept_calls': 0,
        'call_level_rows': 0, 'call_level_unverified': 0, 'call_level_unknown_case': 0,
        'call_level_held_excluded': 0, 'call_level_kept': 0,
    }
    by_site: dict[str, list[dict]] = {}

    for group in groups:
        from ..rl.episode_files import group_rollouts
        path = group_rollouts(group)       # raw or compacted (rollouts.jsonl.zst)
        if path is None:
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
                if imports_native_state(turn['messages']):
                    # a state produced in its episode lives only in that collection's actor store (a live aei1 store
                    # is not read here); the training engine could not render the call, so it is left out and counted
                    funnel['imports_native_state'] += 1
                    continue
                tokens = sum(len(part.get('token_ids') or ()) for part in sampling.get('prompt') or ())
                if tokens > args.max_prompt_tokens:
                    funnel['too_long'] += 1
                    continue
                site = _sha(header)
                key = _sha(json.dumps([rollout['task_id'], rollout.get('sample'), rollout.get('seed'), index]))
                sig_hdr, body_hdr = split_compositional_header(header)
                by_site.setdefault(site, []).append({
                    'site': site, 'binding': site_binding(header), 'header': header,
                    'scope_header': sig_hdr, 'body_header': body_hdr,
                    'key': key, 'source_kind': 'rollout',
                    'task_id': task, 'family': rollout.get('family') or 'default',
                    'source_groups': rollout.get('source_groups') or [],
                    'policy': (rollout.get('policy') or {}).get('checkpoint_sha256'),
                    'messages': turn['messages'], 'tools': turn.get('tools') or [],
                    'target': turn['target'], 'prompt_tokens': tokens,
                    'experience_text': _experience_summary(header, turn['target']),
                })
                funnel['kept_calls'] += 1

    # Engine A verified call-level teacher answers (the target is the teacher's verified answer, never the policy's)
    for path in (getattr(args, 'call_level', None) or ()):
        path = Path(path)
        for r in _jsonl(path / 'call-level.jsonl.zst' if path.is_dir() else path):
            funnel['call_level_rows'] += 1
            if not r.get('verified'):
                funnel['call_level_unverified'] += 1
                continue
            task = r['task_id']
            if task not in cases:
                # its held closure can only be checked over a real case row (call_level_engine checked it at extract;
                # this run re-checks over the collections it was given): fail closed
                funnel['call_level_unknown_case'] += 1
                continue
            if task in excluded:
                funnel['call_level_held_excluded'] += 1
                continue
            header = site_header(r['messages'])
            if header is None:
                funnel['no_header'] += 1
                continue
            if imports_native_state(r['messages']):
                funnel['imports_native_state'] += 1
                continue
            site = _sha(header)
            sig_hdr, body_hdr = split_compositional_header(header)
            by_site.setdefault(site, []).append({
                'site': site, 'binding': site_binding(header), 'header': header,
                'scope_header': sig_hdr, 'body_header': body_hdr,
                'key': _sha(json.dumps(['call_level', r['row_id']])), 'source_kind': 'call_level_teacher',
                'task_id': task, 'family': r.get('family') or 'default', 'source_groups': r.get('source_groups') or [],
                'policy': None, 'teacher': r.get('teacher'), 'messages': r['messages'], 'tools': r.get('tools') or [],
                'target': r['target'], 'prompt_tokens': None,   # no sampled prompt ids: the graph length bounds it
                'experience_text': _experience_summary(header, r['target']),
            })
            funnel['call_level_kept'] += 1
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
    summary = {
        'schema': 'natlang.compiler.t2-rows/2', 'rows': len(rows), 'sites': sites, 'funnel': funnel,
        'splits': {split: sum(1 for row in rows if row['split'] == split) for split in ('train', 'dev-seen', 'dev-unseen')},
        'collections': list(args.collections or ()), 'call_level': [str(p) for p in getattr(args, 'call_level', None) or ()],
        'max_prompt_tokens': args.max_prompt_tokens,
    }
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
    """Each site's text-initialized header state (and its scope/body part states) and each call's graph row, built
    once and cached in RUN/sites.json (the states themselves live in RUN/store)."""

    def __init__(self, engine, run, *, mask_context=True):
        self.engine, self.path, self.mask_context = engine, Path(run) / 'sites.json', mask_context
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {'states': {}}
        self.cache.setdefault('part_states', {})
        self.graphs = {}

    def state(self, row):
        from .compiled_site import init_header_state
        ident = self.cache['states'].get(row['site'])
        if ident is None or self.engine.store.get_native_state(ident) is None:
            ident = init_header_state(self.engine, row['messages'], row['tools'], row['header'])
            self.cache['states'][row['site']] = ident
            _write_json(self.path, self.cache)
        return ident

    def part_block(self, row, text):
        """The text state of one part of the header (its signature line or instructions body), written in the real
        call as the header's own state is (``init_header_state``)."""
        from .compiled_site import init_header_state
        key = _sha(json.dumps([row['site'], text]))
        ident = self.cache['part_states'].get(key)
        if ident is None or self.engine.store.get_native_state(ident) is None:
            ident = init_header_state(self.engine, row['messages'], row['tools'], text)
            self.cache['part_states'][key] = ident
            _write_json(self.path, self.cache)
        return self.engine.store.get_native_state(ident).block

    def block(self, row):
        return self.engine.store.get_native_state(self.state(row)).block

    def row(self, row):
        from .compiled_site import site_graph
        if row['key'] not in self.graphs:
            state = self.state(row)
            block = self.engine.store.get_native_state(state).block
            # a rollout's reply is the policy's own sample; a call-level row's is the teacher's (its empty think block
            # and the token after it are context: serve.targets.target_context_end)
            graph = site_graph(self.engine, row['messages'], row['tools'], row['target'], row['header'], state,
                               len(block.token_ids), graph_id='t2:' + row['key'],
                               source_groups=row['source_groups'] or [row['task_id']],
                               own=row.get('source_kind') == 'rollout', mask_context=self.mask_context)
            self.graphs[row['key']] = {'graph': graph, 'state_id': state}
        return self.graphs[row['key']]


def _lr_scale(step: int, warmup: int, total_steps: int) -> float:
    if step < warmup:
        return float(step + 1) / float(max(1, warmup))
    progress = min(1.0, float(step - warmup) / float(max(1, total_steps - warmup)))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def target_tokens(site):
    """The target weight mass of a site row's graph (its supervised reply tokens)."""
    return float(sum(site['graph']['nodes'][0]['target_weights']))


def row_loss_scales(counts, weighting, cap):
    """Per-row loss scales of one batch (each row's loss is its per-token mean CE). ``tokens``: each row counts by its
    target tokens, at most ``cap`` (0: no cap), so the batch loss is a token mean in which one long trace cannot take
    the whole batch; ``rows``: every row counts equally (a 3-token label as much as a 3,000-token rollout)."""
    if weighting == 'rows':
        return [1. / len(counts)] * len(counts)
    mass = [min(c, cap) if cap else c for c in counts]
    total = sum(mass)
    return [m / total for m in mass] if total > 0 else [1. / len(counts)] * len(counts)


def eval_rng(seed, row):
    """The Live-KB draw of one dev row: seeded by the run seed and the row, so every row sees its own peers and the
    same ones at every eval."""
    return random.Random(_sha(json.dumps([int(seed), row['key']])))


def _means(per_row, arms, rows=None):
    rows = per_row if rows is None else rows
    out = {}
    for arm in arms:
        values = [r[arm] for r in rows if r.get(arm) is not None]
        out[arm] = sum(values) / len(values) if values else None
    return out


def _token_means(per_row, arms):
    out = {}
    for arm in arms:
        pairs = [(r[arm], r['target_tokens']) for r in per_row if r.get(arm) is not None]
        mass = sum(n for _, n in pairs)
        out[arm] = sum(v * n for v, n in pairs) / mass if mass else None
    return out


ABLATION_ARMS = ('base', 'joint', 'kv_only', 'adapter_only', 'mean_core', 'shuffled', 'shuffled_slots', 'shuffled_kb')


def site_decorrelation_loss(z_sites: list[torch.Tensor], *, var_floor: float = 0.5) -> torch.Tensor | None:
    """VICReg-style batch decorrelation over distinct sites' mean slot vectors ``[1, M, d]``.

    Prevents ``HeaderEncoder`` from collapsing distinct sites to a single global vector (observed in run1 where
    pairwise cosine hit 0.998): penalizes squared off-diagonal cosine similarity between distinct sites and keeps
    per-dimension std above ``var_floor``. Returns None when fewer than 2 distinct sites are present."""
    if len(z_sites) < 2:
        return None
    pooled = torch.cat([z.float().mean(dim=1) for z in z_sites], dim=0)   # [K, d]
    normed = torch.nn.functional.normalize(pooled, dim=-1)
    gram = normed @ normed.t()                                            # [K, K]
    k = gram.shape[0]
    off_mask = ~torch.eye(k, dtype=torch.bool, device=gram.device)
    cos_pen = gram[off_mask].pow(2).mean()
    std = torch.sqrt(pooled.var(dim=0, unbiased=False) + 1e-4)
    var_pen = torch.nn.functional.relu(float(var_floor) - std).mean()
    return cos_pen + var_pen


def mean_site_cosine(encoded_slots: dict) -> float | None:
    """Mean off-diagonal cosine similarity between distinct sites' mean slot vectors (collapse diagnostic)."""
    if len(encoded_slots) < 2:
        return None
    vecs = torch.cat([z.detach().float().mean(dim=1) for z, _ in encoded_slots.values()], dim=0)
    normed = torch.nn.functional.normalize(vecs, dim=-1)
    gram = normed @ normed.t()
    off = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    return float(gram[off].mean())


def ablation(engine, head, sites, rows, adapter_id, step, others, *, basis_module=None, writer=None, peer_pool=(),
             seed=0, index=None, k_pos=2, k_neg=2):
    """Mean dev CE per arm, overall, by ``source_kind`` and token-weighted.

    - ``base``: the text state with a zero core (the uncompiled call); ``joint``: the full compiled stack;
      ``kv_only`` / ``adapter_only``: one reader half.
    - ``mean_core``: the site-blind control: no state residual and the adapter-only cores averaged over the dev rows,
      the same for every row. A compiler whose ``adapter_only`` does not beat it has learned one global adapter.
    - ``shuffled``: the whole compiled site of another dev site swapped in (its site slots, slot weights and Live-KB
      slots); ``shuffled_slots``: only its site slots (the row keeps its own slot weights and KB).
    - ``shuffled_kb``: another row's Live-KB slots, from a row whose peers differ from this row's own; rows with no
      such row are listed in ``shuffled_kb_unavailable``.
    A compiler whose real slots do not beat the shuffled ones is a defect (AGENTS.md: a span that ablation cannot tell
    from shuffled state). Each row's KB is drawn by ``eval_rng`` (its own seeded sample, filled by similarity from
    ``index``)."""
    from .compiled_site import init_block_values, site_loss
    from .live_kb_writer import build_live_kb_slots
    head.eval()
    if writer is not None:
        writer.eval()
    per_row = []
    with torch.no_grad():
        encoded = {}
        for row in others:
            if row['site'] not in encoded:
                z, _, w, _ = head.encode(init_block_values(sites.block(row)))
                encoded[row['site']] = (z, w)
        kb, peers = {}, {}
        if writer is not None:
            for row in rows:
                chosen = []
                kb[row['key']] = build_live_kb_slots(engine, writer, row, list(peer_pool), k_pos=k_pos, k_neg=k_neg,
                                                     rng=eval_rng(seed, row), index=index, chosen=chosen)
                peers[row['key']] = frozenset(chosen)
        kb_args_of = {key: {'z_kb': z, 'k_kb': k, 'w_kb': w, 'kb_valid': v} for key, (z, k, w, v) in kb.items()}
        # the site-blind core: adapter-only cores of every dev row, averaged
        cores = [head(init_block_values(sites.block(row)), kv=False, apply_dropout=False,
                      **kb_args_of.get(row['key'], {})).cores for row in rows]
        mean_core = {name: sum(c[name] for c in cores) / len(cores) for name in cores[0]} if cores else None
    unavailable = []
    for index_row, row in enumerate(rows):
        site = sites.row(row)
        block = sites.block(row)
        rotation = others[index_row + 1:] + others[:index_row + 1]
        other = next((o for o in rotation if o['site'] != row['site']), None)
        kb_args = kb_args_of.get(row['key'], {})
        common = dict(adapter_id=adapter_id, init_block=block, basis_module=basis_module, apply_dropout=False)
        losses = {
            'base': site_loss(engine, head, site, kv=False, adapter=False, **common),
            'joint': site_loss(engine, head, site, **common, **kb_args),
            'kv_only': site_loss(engine, head, site, adapter=False, **common, **kb_args),
            'adapter_only': site_loss(engine, head, site, kv=False, **common, **kb_args),
        }
        if mean_core is not None:
            losses['mean_core'] = site_loss(engine, head, site, kv=False, adapter=False, extra_cores=mean_core, **common)
        if other is not None:
            z_other, w_other = encoded[other['site']]
            losses['shuffled_slots'] = site_loss(engine, head, site, z_slots=z_other, **common, **kb_args)
            losses['shuffled'] = site_loss(engine, head, site, z_slots=z_other, slot_weights=w_other, **common,
                                           **kb_args_of.get(other['key'], {}))
        if kb_args:
            kb_other = next((o for o in rotation if o['key'] != row['key'] and o['key'] in kb
                             and kb[o['key']][0].shape[1] > 0 and peers[o['key']] != peers[row['key']]), None)
            if kb_other is None:
                unavailable.append(row['key'])
            else:
                losses['shuffled_kb'] = site_loss(engine, head, site, **common, **kb_args_of[kb_other['key']])
        per_row.append({'key': row['key'], 'site': row['site'], 'split': row['split'],
                        'source_kind': row.get('source_kind'), 'target_tokens': target_tokens(site),
                        'kb_peers': sorted(peers.get(row['key'], ())), **losses})
    head.train()
    if writer is not None:
        writer.train()
    mean = _means(per_row, ABLATION_ARMS)

    def paired(arm):
        rows_ = [r for r in per_row if r.get(arm) is not None]
        return rows_, (sum(r[arm] - r['joint'] for r in rows_) / len(rows_) if rows_ else None)

    out = {'step': step, 'rows': len(rows), 'mean_ce': mean,
           'mean_ce_by_source': {kind: _means(per_row, ABLATION_ARMS, [r for r in per_row if (r['source_kind'] or 'unknown') == kind])
                                 for kind in sorted({r['source_kind'] or 'unknown' for r in per_row})},
           'mean_ce_tokens': _token_means(per_row, ABLATION_ARMS),
           'base_minus_joint': None if mean['joint'] is None else mean['base'] - mean['joint'],
           'z_site_mean_cos': mean_site_cosine(encoded),
           'shuffled_kb_unavailable': unavailable}
    for arm in ('shuffled', 'shuffled_slots', 'shuffled_kb', 'mean_core'):
        rows_, delta = paired(arm)
        out[f'{arm}_minus_joint'] = delta
        out[f'joint_beats_{arm}'] = f"{sum(1 for r in rows_ if r['joint'] < r[arm])}/{len(rows_)}"
    rows_, _ = paired('mean_core')
    out['adapter_only_beats_mean_core'] = f"{sum(1 for r in rows_ if r['adapter_only'] < r['mean_core'])}/{len(rows_)}"
    out['per_row'] = per_row
    return out


def _frozen(*modules):
    """Context: the modules' parameters take no gradient (an inner loop that differentiates only its own leaves)."""
    import contextlib

    @contextlib.contextmanager
    def scope():
        saved = [(p, p.requires_grad) for module in modules if module is not None for p in module.parameters()]
        for p, _ in saved:
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, flag in saved:
                p.requires_grad_(flag)
    return scope()


def _fo_maml_episode_step(engine, head, sites, site_calls, *, adapter_id, basis_module, inner_steps, inner_lr, prox_lambda,
                          reptile_weight, scale):
    """One FO-MAML + Reptile episode over a multi-call site (``plans/NATLANG_COMPILER.md`` §9.4 part 4).

    Support = every call but the last, query = the last. The inner loop runs SGD on ``z_site`` alone (head and basis
    frozen) over the mean support loss with the proximal pull; the query loss is taken at ``z0 + sg(z_K - z0)``, so its
    gradient reaches the header encoder through ``z0`` and the head and basis directly (first-order meta-gradient), and
    a Reptile term pulls the encoder's ``z0`` toward ``z_K``. Returns the query loss."""
    from .compiled_site import init_block_values, site_loss
    import random as _rnd
    q_idx = _rnd.randrange(len(site_calls))
    query = site_calls[q_idx]
    support = [c for i, c in enumerate(site_calls) if i != q_idx]
    block = sites.block(query)
    z0 = head.encode(init_block_values(block))[0]
    with _frozen(head, basis_module):
        z_ref = z0.detach().float()
        z = z_ref.clone().requires_grad_(True)
        for _ in range(int(inner_steps)):
            z.grad = None
            for row in support:
                site_loss(engine, head, sites.row(row), adapter_id=adapter_id, init_block=block, z_slots=z,
                          basis_module=basis_module, apply_dropout=False, backward=True, scale=1. / len(support))
            if prox_lambda > 0.0:
                (0.5 * float(prox_lambda) * (z - z_ref).pow(2).sum()).backward()
            if z.grad is None:
                break
            torch.nn.utils.clip_grad_norm_([z], 1.0)
            z = (z.detach() - float(inner_lr) * z.grad).requires_grad_(True)
        z_k = z.detach()
    z_meta = z0 + (z_k - z0.detach())
    q_loss = site_loss(engine, head, sites.row(query), adapter_id=adapter_id, init_block=block, z_slots=z_meta,
                       basis_module=basis_module, apply_dropout=False, backward=True, scale=scale)
    if reptile_weight > 0.0:
        # the query backward (inside site_loss) freed z0's graph: the Reptile pull re-encodes the header (a small
        # encoder forward) rather than backpropagating through z0 twice (the first version raised here)
        z0_again = head.encode(init_block_values(block))[0]
        (float(reptile_weight) * (z0_again - z_k).pow(2).mean() * scale).backward()
    return q_loss


def _compositional_step(engine, head, sites, row, *, adapter_id, basis_module, scale, kb_args=None):
    """One call compiled as ``C = C_scope + C_body`` (``plans/NATLANG_COMPILER.md`` §4.4): the signature line and the
    instructions body each compiled alone (their text states written in the real call), their cores summed in the
    shared basis, the state residual from the full header. Teaches the head cores that compose. None when the header
    has no body or a part does not occur exactly once in the call."""
    from .compiled_site import init_block_values, site_loss
    if not row['body_header']:
        return None
    try:
        parts = [sites.part_block(row, text) for text in (row['scope_header'], row['body_header'])]
    except ValueError:
        return None
    c_scope, c_body = (head(init_block_values(part), kv=False, apply_dropout=False).cores for part in parts)
    from .compiled_site import compose_site_cores
    return site_loss(engine, head, sites.row(row), adapter_id=adapter_id, init_block=sites.block(row),
                     basis_module=basis_module, adapter=False, extra_cores=compose_site_cores(c_scope, c_body),
                     apply_dropout=False, backward=True, scale=scale, **(kb_args or {}))


def _em_entry_drift_and_repro(
    engine,
    head,
    writer,
    repro_optimizer,
    em_kb,
    sites,
    peer_clusters: list[list[dict]],
    *,
    adapter_id: str,
    basis_module,
    rng: random.Random,
    entry_steps: int = 8,
    entry_lr: float = 0.05,
    prox_lambda: float = 0.05,
    repro_steps: int = 8,
    repro_key_weight: float = 0.5,
) -> dict:
    """Mode-2 EM cycle inside ``run_t2``: E-step (direct KB entry drift with held-query gate) -> M-step (reproduction)."""
    from .compiled_site import site_loss
    from .hkb_bridge import ExperienceEntry, record_key
    from .hkb_em import ReproductionConfig, reproduction_step

    e_accepted = False
    e_query_gain = 0.0
    e_drift_norm = 0.0

    if peer_clusters:
        cluster = rng.choice(peer_clusters)
        # Partition by task so source, support, and query are strictly disjoint tasks (zero self-leakage).
        by_task: dict[str, list[dict]] = {}
        for row in cluster:
            by_task.setdefault(str(row['task']), []).append(row)
        task_ids = list(by_task.keys())
        if len(task_ids) >= 3:
            rng.shuffle(task_ids)
            src_row = by_task[task_ids[0]][0]
            sup_row = by_task[task_ids[1]][0]
            qry_row = by_task[task_ids[2]][0]
            src_text = _experience_summary(src_row['header'], src_row['target'])
            scope = f"fn:{src_row['site'][:12]}"
            rid = src_row['key']
            fetched = em_kb.kb.fetch_records([record_key(scope, rid)], scope=scope)
            device = next(head.parameters()).device
            if fetched:
                z0 = fetched[0].value.detach().float().to(device)
                k0 = fetched[0].key.detach().float().to(device)
            else:
                with torch.no_grad():
                    z_init, k_init = writer.encode_experience_texts(engine, [src_text])
                z0 = z_init[0].detach().float().to(device)
                k0 = k_init[0].detach().float().to(device)
                em_kb.add(ExperienceEntry(
                    record_id=rid, scope=scope, key=k0.cpu(), value=z0.cpu(),
                    text=src_text, task_ids=(str(src_row['task']),), strength=1.0,
                ))

            # E-step: freeze head/basis/writer, optimize ONLY z_entry on support peer, gate strictly on query peer.
            with _frozen(head, basis_module, writer):
                w_kb = torch.ones((1, 1), dtype=torch.float32, device=device)
                v_kb = torch.ones((1, 1), dtype=torch.bool, device=device)
                k_fixed = k0.view(1, 1, -1)
                with torch.no_grad():
                    base_q = site_loss(
                        engine, head, sites.row(qry_row), adapter_id=adapter_id, init_block=sites.block(qry_row),
                        basis_module=basis_module, z_kb=z0.view(1, 1, -1), k_kb=k_fixed, w_kb=w_kb, kb_valid=v_kb,
                        backward=False,
                    )
                z_opt = z0.clone().requires_grad_(True)
                z_optim = torch.optim.AdamW([z_opt], lr=entry_lr, weight_decay=0.0)
                for _ in range(max(1, int(entry_steps))):
                    z_optim.zero_grad(set_to_none=True)
                    site_loss(
                        engine, head, sites.row(sup_row), adapter_id=adapter_id, init_block=sites.block(sup_row),
                        basis_module=basis_module, z_kb=z_opt.view(1, 1, -1), k_kb=k_fixed, w_kb=w_kb, kb_valid=v_kb,
                        backward=True,
                    )
                    if prox_lambda > 0.0:
                        (0.5 * float(prox_lambda) * (z_opt - z0).pow(2).sum()).backward()
                    z_optim.step()
                with torch.no_grad():
                    new_q = site_loss(
                        engine, head, sites.row(qry_row), adapter_id=adapter_id, init_block=sites.block(qry_row),
                        basis_module=basis_module, z_kb=z_opt.view(1, 1, -1), k_kb=k_fixed, w_kb=w_kb, kb_valid=v_kb,
                        backward=False,
                    )
                e_query_gain = float(base_q - new_q)
                if e_query_gain > 1e-4:
                    upd = em_kb.update_entry_payload(scope, rid, new_value=z_opt.detach().cpu(), mark_drifted=True)
                    e_accepted = True
                    e_drift_norm = float(upd['drift_norm'])

    # M-step: train hindsight encoder to reproduce all unreproduced drifted entries on disk.
    m_res = reproduction_step(
        writer, engine, repro_optimizer, em_kb,
        cfg=ReproductionConfig(steps=int(repro_steps), key_weight=float(repro_key_weight), mark_after=True),
    )
    return {
        'e_accepted': e_accepted,
        'e_query_gain': round(e_query_gain, 6),
        'e_drift_norm': round(e_drift_norm, 6),
        'm_entries': int(m_res['entries']),
        'm_initial_loss': round(float(m_res['initial_loss']), 6),
        'm_final_loss': round(float(m_res['final_loss']), 6),
        'm_marked': int(m_res['marked']),
    }


def train(args):
    from ..common.code_provenance import code_provenance
    from ..common.hashing import sha256_file_hex
    from ..model.native_adapters import target_parameters
    from ..serve import load_native_engine
    from ..serve.store import TensorStore
    from .compiled_site import (
        CompiledSiteConfig, CompiledSiteHead, compile_release, create_compiler_adapter,
        create_learned_compiler_basis, site_loss, state_layout,
    )
    from .live_kb_writer import (KbIndex, LiveExperienceWriter, PplxWarmstartWriter, _engine_embedding_weight,
                                 build_live_kb_slots)

    run = Path(args.out)
    run.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = list(_jsonl(args.rows))
    train_rows = [row for row in rows if row['split'] == 'train']
    dev_rows = [row for row in rows if row['split'] != 'train']
    rng = random.Random(args.seed)
    dev_rows = rng.sample(dev_rows, min(len(dev_rows), args.dev_rows)) if dev_rows else []

    # multi-call training sites: the FO-MAML episodes
    train_by_site: dict[str, list[dict]] = {}
    for row in train_rows:
        train_by_site.setdefault(row['site'], []).append(row)
    maml_sites = [calls for calls in train_by_site.values() if len(calls) >= 2]

    engine = load_native_engine(str(args.checkpoint), base=str(args.base), device=args.device,
                                dtype=getattr(torch, args.dtype), training_state=False)
    engine.store = TensorStore(run / 'store')
    sites = Sites(engine, run, mask_context=args.mask_non_student_context)
    state_path = run / 'learner-state.pt'
    resumed = state_path.exists()
    saved = torch.load(state_path, map_location='cpu', weights_only=False) if resumed else None
    if saved is not None and saved.get('objective') != OBJECTIVE:
        raise ValueError(f"{state_path} was trained under objective {saved.get('objective')!r}, not {OBJECTIVE!r}: "
                         'start a new run directory')

    if saved is not None:
        config = CompiledSiteConfig.from_dict(saved['config'])
        adapter_id = saved['adapter_id']
        learn_basis = bool(saved.get('learn_basis', False))
    else:
        targets = tuple(sorted(target_parameters(engine.shared_model)))
        config = CompiledSiteConfig(
            layout=tuple(state_layout(sites.block(train_rows[0]))),
            targets=targets,
            rank=args.rank,
            adapter_inner_rank=args.inner_rank,
            kv_drop=args.kv_drop,
            adapter_drop=args.adapter_drop,
            state_residual_scale=getattr(args, 'state_residual_scale', 0.1),
        )
        learn_basis = bool(args.learn_basis)
        adapter_id = None

    basis_module = None
    if learn_basis:
        basis_module = create_learned_compiler_basis(engine, config.targets, config.rank)
        if saved is not None and saved.get('basis') is not None:
            basis_module.load_state_dict(saved['basis'])
            basis_id, adapter_id = basis_module.save_artifact(engine)
            basis_module.basis_id = basis_id
            basis_module.adapter_id = adapter_id
        else:
            adapter_id = basis_module.adapter_id
        basis_module = basis_module.to(args.device)
    elif adapter_id is None:
        adapter_id = create_compiler_adapter(engine, config.targets, config.rank)

    if engine.store.get_native_tensor_artifact(adapter_id) is None:
        raise ValueError('the run store lost its compiler adapter basis; this run cannot resume')

    head = CompiledSiteHead(config).to(args.device)
    writer = None
    writer_backend = getattr(args, 'writer_backend', 'pplx')
    warmstart_dir = getattr(args, 'warmstart_dir', None)
    if warmstart_dir is not None and writer_backend != 'pplx':
        raise ValueError('--warmstart-dir requires --writer-backend pplx (000000002 weights belong to pplx-embed-v2-late-0.6b)')
    if args.live_kb:
        if writer_backend == 'pplx':
            pplx_rev = getattr(args, 'pplx_revision', None)
            if not pplx_rev:
                raise ValueError('--writer-backend pplx requires --pplx-revision (pinned HF commit SHA)')
            if warmstart_dir is None and not resumed:
                raise ValueError('--writer-backend pplx starts from 000000002: pass --warmstart-dir (writer_H.pt, query_Q.pt)')
            writer = PplxWarmstartWriter.from_warmstart(
                revision=pplx_rev,
                warmstart_dir=None if resumed else str(warmstart_dir),
                value_dim=config.value_dim,
                key_dim=config.key_dim,
                device=args.device,
                dtype=getattr(torch, args.dtype),
                use_hf_kernels=getattr(args, 'pplx_hf_kernels', None),
            )
        else:
            # owner 2026-11-26: Stage C never trains on the embedding stand-in; it exists for CPU unit tests
            if str(args.device) != 'cpu':
                raise ValueError('--writer-backend embed is the CPU unit-test stand-in; Stage C trains with --writer-backend pplx')
            embed_dim = int(_engine_embedding_weight(engine).shape[-1])
            writer = LiveExperienceWriter(embed_dim, value_dim=config.value_dim, key_dim=config.key_dim).to(args.device)

    # Dual-LR optimizer groups: fresh head/H_val at --lr (2e-4), warm-started 000000002 H_key/Q at --warm-lr (2e-5), basis at --basis-lr
    param_groups = [{'params': list(head.parameters()), 'lr': args.lr, 'base_lr': args.lr, 'name': 'head'}]
    if basis_module is not None:
        param_groups.append({'params': list(basis_module.parameters()), 'lr': args.basis_lr, 'base_lr': args.basis_lr, 'name': 'basis'})
    if writer is not None:
        fresh_w = writer.fresh_parameters()
        warm_w = writer.warm_parameters()
        if fresh_w:
            param_groups.append({'params': fresh_w, 'lr': args.lr, 'base_lr': args.lr, 'name': 'writer_fresh'})
        if warm_w:
            warm_lr = getattr(args, 'warm_lr', 2e-5)
            param_groups.append({'params': warm_w, 'lr': warm_lr, 'base_lr': warm_lr, 'name': 'writer_warm'})
    # FP32 master weights and moments for every parameter (AGENTS.md "Precision"): the pplx encoders' parameters are
    # BF16, so a plain AdamW would round each 2e-5 update away; MasterAdamW is plain AdamW on FP32 parameters.
    from .hkb_dependency import require as _require_hkb
    _require_hkb()
    from hindsight_kb.recipe.optimizer import MasterAdamW
    optimizer = MasterAdamW(param_groups, weight_decay=args.weight_decay, betas=(.9, .95))

    step, order, position = 0, list(range(len(train_rows))), 0
    if saved is not None:
        head.load_state_dict(saved['head'])
        if writer is not None and saved.get('writer') is not None:
            writer.load_state_dict(saved['writer'])
        optimizer.load_state_dict(saved['optimizer'])
        step, order, position = saved['step'], saved['order'], saved['position']
        torch.set_rng_state(saved['rng']['torch'])
        random.setstate(saved['rng']['python'])
        rng.setstate(saved['rng']['run'])   # Live-KB peers, null rows, episode and compositional draws
    else:
        rng.shuffle(order)

    all_params = [p for g in param_groups for p in g['params']]
    receipt = {
        'schema': RECEIPT_SCHEMA, 'code': code_provenance(), 'hindsight_kb': _hkb_provenance(),
        'engine_identity': dict(engine.native_identity), 'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': args.checkpoint_sha256, 'base': str(args.base), 'rows': str(args.rows),
        'rows_sha256': sha256_file_hex(args.rows), 'rows_summary': _summary(args.rows), 'config': config.to_dict(),
        'adapter_id': adapter_id, 'learn_basis': learn_basis, 'live_kb': bool(args.live_kb),
        'maml_sites': len(maml_sites),
        'args': {key: str(value) for key, value in vars(args).items() if key != 'func'},
        'parameters': sum(p.numel() for p in all_params),
        'parameter_dtypes': sorted({str(p.dtype) for p in all_params}),
        'resumed_at_step': step if resumed else None,
        'objective': OBJECTIVE,
        'writer_backend': writer_backend if writer is not None else None,
        'writer_warmstart': getattr(writer, 'warmstart', None),
        'target_context': 'masked' if args.mask_non_student_context else 'trained',
        'loss_weighting': {'kind': args.loss_weighting, 'row_token_cap': args.row_token_cap},
        'kb_rule': {'index_every': args.kb_index_every, 'neg_grad': args.kb_neg_grad},
    }
    _write_json(run / ('receipt.json' if not resumed else f'receipt-resume-{step}.json'), receipt)
    log = (run / 'train.jsonl').open('a', encoding='utf-8')
    abl = (run / 'ablation.jsonl').open('a', encoding='utf-8')

    # the writer's keys of the train pool: positives are filled by similarity from it (rebuilt on a resume from the
    # restored writer, refreshed every --kb-index-every steps)
    index = KbIndex(engine, writer, train_rows) if writer is not None and args.kb_index_every > 0 else None
    em_every = int(getattr(args, 'em_every', 0) or 0)
    em_kb = None
    repro_optimizer = None
    em_peer_clusters: list[list[dict]] = []
    if writer is not None and em_every > 0:
        from .hkb_bridge import HeldAdmission, ScopedExperienceKB
        em_kb_path = getattr(args, 'em_kb', None) or (run / 'em-kb.sqlite')
        train_task_ids = frozenset(str(r['task']) for r in train_rows)
        em_kb = ScopedExperienceKB(
            em_kb_path,
            admission=HeldAdmission(known=train_task_ids, excluded=frozenset()),
            key_dim=config.key_dim,
            value_dim=config.value_dim,
        )
        repro_groups = []
        if writer.fresh_parameters():
            repro_groups.append({'params': writer.fresh_parameters(), 'lr': args.lr})
        if writer.warm_parameters():
            repro_groups.append({'params': writer.warm_parameters(), 'lr': getattr(args, 'warm_lr', 2e-5)})
        repro_optimizer = MasterAdamW(repro_groups, weight_decay=args.weight_decay, betas=(.9, .95))
        by_fam: dict[str, list[dict]] = {}
        for r in train_rows:
            by_fam.setdefault(str(r.get('family') or r['site']), []).append(r)
        em_peer_clusters = [rows for rows in by_fam.values() if len({str(r['task']) for r in rows}) >= 3]

    abl_args = dict(basis_module=basis_module, writer=writer, peer_pool=train_rows, seed=args.seed, index=index,
                    k_pos=args.kb_pos, k_neg=args.kb_neg)
    if step == 0 and not resumed and dev_rows:
        result = ablation(engine, head, sites, dev_rows, adapter_id, 0, dev_rows, **abl_args)
        abl.write(json.dumps(result) + '\n'); abl.flush()

    head.train()
    if writer is not None:
        writer.train()
    while step < args.steps:
        started = time.time()
        lr_mult = _lr_scale(step, args.warmup, args.steps)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * lr_mult
        optimizer.zero_grad(set_to_none=True)
        losses, tokens = [], 0
        maml_loss_val, comp_loss_val = None, None

        batch_rows = []
        for _ in range(args.batch):
            if position == len(order):
                rng.shuffle(order)
                position = 0
            batch_rows.append(train_rows[order[position]])
            position += 1
        batch_sites = [sites.row(row) for row in batch_rows]
        scales = row_loss_scales([target_tokens(site) for site in batch_sites], args.loss_weighting, args.row_token_cap)
        for row, site, row_scale in zip(batch_rows, batch_sites, scales):
            tokens += len(site['graph']['nodes'][0]['token_ids'])
            kb_args = {}
            if writer is not None:
                null_step = rng.random() < args.null_kb_prob
                z_kb, k_kb, w_kb, v_kb = build_live_kb_slots(
                    engine, writer, row, train_rows, k_pos=args.kb_pos, k_neg=args.kb_neg,
                    null_restraint=null_step, rng=rng, index=index, neg_grad=args.kb_neg_grad,
                )
                kb_args = {'z_kb': z_kb, 'k_kb': k_kb, 'w_kb': w_kb, 'kb_valid': v_kb}
            losses.append(site_loss(
                engine, head, site, adapter_id=adapter_id, init_block=sites.block(row),
                basis_module=basis_module, **kb_args, backward=True, scale=row_scale,
            ))

        # Site-distinctness regularizer across distinct sites in the batch (forbids z_site collapse to one global adapter)
        site_var_loss_val = None
        site_var_weight = float(getattr(args, 'site_var_weight', 0.05))
        if site_var_weight > 0.0:
            from .compiled_site import init_block_values
            seen_sites = {}
            for row in batch_rows:
                if row['site'] not in seen_sites:
                    seen_sites[row['site']] = head.encode(init_block_values(sites.block(row)))[0]
            reg = site_decorrelation_loss(list(seen_sites.values()))
            if reg is not None:
                (reg * site_var_weight).backward()
                site_var_loss_val = float(reg.detach())

        # FO-MAML episode: z_site meta-trained for few-shot Tier-1.5 refinement
        if args.meta_every > 0 and maml_sites and (step + 1) % args.meta_every == 0:
            episode_calls = rng.choice(maml_sites)
            maml_loss_val = _fo_maml_episode_step(
                engine, head, sites, episode_calls, adapter_id=adapter_id, basis_module=basis_module,
                inner_steps=args.meta_inner_steps, inner_lr=args.meta_inner_lr,
                prox_lambda=args.meta_prox_lambda, reptile_weight=args.reptile_weight,
                scale=args.meta_weight,
            )

        # compositional cores: C_scope + C_body (jointly with Live-KB when active)
        if args.comp_every > 0 and (step + 1) % args.comp_every == 0:
            comp_row = rng.choice(train_rows)
            comp_kb = {}
            if writer is not None:
                cz, ck, cw, cv = build_live_kb_slots(
                    engine, writer, comp_row, train_rows, k_pos=args.kb_pos, k_neg=args.kb_neg,
                    null_restraint=(rng.random() < args.null_kb_prob), rng=rng, index=index,
                    neg_grad=args.kb_neg_grad,
                )
                comp_kb = {'z_kb': cz, 'k_kb': ck, 'w_kb': cw, 'kb_valid': cv}
            comp_loss_val = _compositional_step(
                engine, head, sites, comp_row, adapter_id=adapter_id, basis_module=basis_module,
                scale=args.comp_weight, kb_args=comp_kb,
            )

        grad_norm = float(torch.nn.utils.clip_grad_norm_(all_params, args.clip))
        optimizer.step()
        step += 1

        em_report = None
        if em_kb is not None and repro_optimizer is not None and em_every > 0 and step % em_every == 0:
            em_report = _em_entry_drift_and_repro(
                engine, head, writer, repro_optimizer, em_kb, sites, em_peer_clusters,
                adapter_id=adapter_id, basis_module=basis_module, rng=rng,
                entry_steps=int(getattr(args, 'em_entry_steps', 8)),
                entry_lr=float(getattr(args, 'em_entry_lr', 0.05)),
                prox_lambda=float(getattr(args, 'em_prox_lambda', 0.05)),
                repro_steps=int(getattr(args, 'em_repro_steps', 8)),
                repro_key_weight=float(getattr(args, 'em_repro_key_weight', 0.5)),
            )

        encoder_grad = sum(float(p.grad.norm()) ** 2 for p in head.encoder.parameters() if p.grad is not None) ** .5
        basis_grad = (
            sum(float(p.grad.norm()) ** 2 for p in basis_module.parameters() if p.grad is not None) ** .5
            if basis_module is not None else 0.0
        )
        writer_grad = (
            sum(float(p.grad.norm()) ** 2 for p in writer.parameters() if p.grad is not None) ** .5
            if writer is not None else 0.0
        )
        log.write(json.dumps({
            'step': step, 'loss': sum(losses) / len(losses), 'losses': losses,
            'loss_tokens': sum(l * c for l, c in zip(losses, scales)), 'row_scales': scales,
            'site_var_loss': site_var_loss_val,
            'maml_loss': maml_loss_val, 'comp_loss': comp_loss_val,
            'em': em_report,
            'grad_norm': grad_norm, 'encoder_grad_norm': encoder_grad,
            'basis_grad_norm': basis_grad, 'writer_grad_norm': writer_grad,
            'lr': optimizer.param_groups[0]['lr'], 'tokens': tokens,
            'seconds': round(time.time() - started, 3),
            'cuda_peak_gib': (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else None,
        }) + '\n')
        log.flush()

        if step % args.state_every == 0 or step == args.steps:
            _atomic_save({
                'head': head.state_dict(),
                'basis': None if basis_module is None else basis_module.state_dict(),
                'writer': None if writer is None else writer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step, 'order': order, 'position': position,
                'config': config.to_dict(), 'adapter_id': adapter_id, 'learn_basis': learn_basis,
                'objective': OBJECTIVE,
                'rng': {'torch': torch.get_rng_state(), 'python': random.getstate(), 'run': rng.getstate()},
            }, state_path)
        if index is not None and step % args.kb_index_every == 0:
            index = abl_args['index'] = KbIndex(engine, writer, train_rows)
        if dev_rows and (step % args.eval_every == 0 or step == args.steps):
            result = ablation(engine, head, sites, dev_rows, adapter_id, step, dev_rows, **abl_args)
            abl.write(json.dumps(result) + '\n'); abl.flush()

    release_row = next((row for row in dev_rows if row['split'] == 'dev-seen'), dev_rows[0] if dev_rows else train_rows[0])
    compile_release(
        engine, head, adapter_id=adapter_id, init_block=sites.block(release_row),
        basis_module=basis_module, out=run / 'release',
        site_id=release_row['site'],
        provenance={'receipt': receipt['code'], 'step': step, 'row': release_row['key'], 'header': release_row['header']},
    )
    (run / 'release' / 'site-row.json').write_text(json.dumps(release_row))
    if em_kb is not None:
        em_kb.close()
    return 0


def export_release(args):
    """Export a Mode-P compiled-site release from any saved T2 ``learner-state.pt`` without waiting for step 5,000."""
    from ..common.code_provenance import code_provenance
    from ..serve import load_native_engine
    from ..serve.store import TensorStore
    from .compiled_site import (
        CompiledSiteConfig, CompiledSiteHead, compile_release, create_compiler_adapter,
        create_learned_compiler_basis,
    )
    saved = torch.load(args.state, map_location='cpu', weights_only=False)
    config = CompiledSiteConfig.from_dict(saved['config'])
    rows = list(_jsonl(args.rows))
    row = (next((r for r in rows if r.get('site') == args.site), None)
           if args.site else next((r for r in rows if r.get('split') == 'dev-seen'), rows[0]))
    if row is None:
        raise ValueError(f'site {args.site!r} not found in {args.rows}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = load_native_engine(str(args.checkpoint), base=str(args.base), device=args.device,
                                dtype=getattr(torch, args.dtype), training_state=False)
    engine.store = TensorStore(out / 'store')
    sites = Sites(engine, out)
    basis_module = None
    if saved.get('learn_basis') and saved.get('basis') is not None:
        basis_module = create_learned_compiler_basis(engine, config.targets, config.rank)
        basis_module.load_state_dict(saved['basis'])
        basis_id, adapter_id = basis_module.save_artifact(engine)
        basis_module.basis_id, basis_module.adapter_id = basis_id, adapter_id
        basis_module = basis_module.to(args.device)
    else:
        adapter_id = create_compiler_adapter(engine, config.targets, config.rank)
    head = CompiledSiteHead(config).to(args.device)
    head.load_state_dict(saved['head'])
    head.eval()
    result = compile_release(
        engine, head, adapter_id=adapter_id, init_block=sites.block(row),
        basis_module=basis_module, out=out, site_id=row['site'],
        provenance={'code': code_provenance(), 'step': saved.get('step'), 'source_state': str(args.state),
                    'row': row['key'], 'header': row['header']},
    )
    (out / 'site-row.json').write_text(json.dumps(row))
    print(json.dumps({'out': str(out), 'site': row['site'], 'step': saved.get('step'),
                      'softStates': result.get('softStates'), 'artifacts': result.get('artifacts')}, indent=2))
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
    p.add_argument('--collections', nargs='*', default=[], help='collection directories (each with groups/g*/)')
    p.add_argument('--call-level', nargs='*', default=[], help='call_level_engine teach outputs (DIR or its call-level.jsonl.zst)')
    p.add_argument('--cases', nargs='*', default=[], help='more case rows for call-level tasks (case JSONL, or an authored pool\'s authored.jsonl); held closure is checked over them')
    p.add_argument('--out', required=True)
    p.add_argument('--max-prompt-tokens', type=int, default=8192)
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
    t.add_argument('--steps', type=int, default=5000)
    t.add_argument('--batch', type=int, default=4)
    t.add_argument('--lr', type=float, default=2e-4)
    t.add_argument('--warm-lr', type=float, default=2e-5)
    t.add_argument('--basis-lr', type=float, default=1e-4)
    t.add_argument('--weight-decay', type=float, default=.01)
    t.add_argument('--warmup', type=int, default=200)
    t.add_argument('--clip', type=float, default=1.)
    t.add_argument('--rank', type=int, default=16)
    t.add_argument('--inner-rank', type=int, default=4)
    t.add_argument('--kv-drop', type=float, default=0.1)
    t.add_argument('--adapter-drop', type=float, default=0.1)
    t.add_argument('--state-residual-scale', type=float, default=0.1,
                   help='Initial scale of tanh-bounded KV/conv state residual (0: legacy unbounded residual).')
    t.add_argument('--site-var-weight', type=float, default=0.05,
                   help='Batch site-decorrelation/variance regularization weight preventing z_site collapse (0 opts out).')
    t.add_argument('--learn-basis', action=argparse.BooleanOptionalAction, default=True)
    t.add_argument('--live-kb', action=argparse.BooleanOptionalAction, default=True)
    t.add_argument('--writer-backend', choices=('pplx', 'embed'), default='pplx',
                   help='pplx: hindsight-kb PPLXBlockEncoder H/Q warm-started from 000000002 (Stage C); embed: CPU unit-test stand-in')
    t.add_argument('--warmstart-dir', type=Path, default=None, help='directory containing writer_H.pt and query_Q.pt from extract_lfm3_warmstart.py')
    t.add_argument('--pplx-revision', default=None, help='pinned HF commit SHA for perplexity-ai/pplx-embed-v2-late-0.6b')
    t.add_argument('--pplx-hf-kernels', action=argparse.BooleanOptionalAction, default=None,
                   help="the pplx backbone's FLA/causal-conv/Liger kernels (default: on for CUDA, as 000000002 trained)")
    t.add_argument('--kb-pos', type=int, default=2)
    t.add_argument('--kb-neg', type=int, default=2)
    t.add_argument('--kb-index-every', type=int, default=250,
                   help="refresh the writer's key index of the train pool every N steps; positives are filled by "
                        'similarity from it where a row has too few same-family/same-site peers (0: no fill)')
    t.add_argument('--kb-neg-grad', action=argparse.BooleanOptionalAction, default=True,
                   help='encode Live-KB distractors with gradient into the writer (--no-kb-neg-grad: without)')
    t.add_argument('--mask-non-student-context', action=argparse.BooleanOptionalAction, default=True,
                   help="a teacher reply's think block (empty or not) and lead-in, and an own reply's unsampled empty "
                        'think block, are zero-weight context (serve.targets.target_context_end)')
    t.add_argument('--loss-weighting', choices=('tokens', 'rows'), default='tokens',
                   help='tokens: each row counts by its target tokens (capped by --row-token-cap); rows: equally')
    t.add_argument('--row-token-cap', type=int, default=512,
                   help='with --loss-weighting tokens, a row counts as at most this many target tokens (0: no cap)')
    t.add_argument('--null-kb-prob', type=float, default=0.2, help='rows that see distractors only (plan: 20%%)')
    t.add_argument('--meta-every', type=int, default=4)
    t.add_argument('--meta-inner-steps', type=int, default=2)
    t.add_argument('--meta-inner-lr', type=float, default=0.05)
    t.add_argument('--meta-prox-lambda', type=float, default=0.01)
    t.add_argument('--meta-weight', type=float, default=0.5)
    t.add_argument('--reptile-weight', type=float, default=0.1)
    t.add_argument('--comp-every', type=int, default=4)
    t.add_argument('--comp-weight', type=float, default=0.5)
    t.add_argument('--em-every', type=int, default=16,
                   help='alternating Mode-2 EM cycle (direct KB entry drift + encoder reproduction) every N steps (0: off)')
    t.add_argument('--em-kb', type=Path, default=None, help='optional ScopedExperienceKB SQLite path (default: RUN/em-kb.sqlite)')
    t.add_argument('--em-entry-steps', type=int, default=8)
    t.add_argument('--em-entry-lr', type=float, default=0.05)
    t.add_argument('--em-prox-lambda', type=float, default=0.05)
    t.add_argument('--em-repro-steps', type=int, default=8)
    t.add_argument('--em-repro-key-weight', type=float, default=0.5)
    t.add_argument('--dev-rows', type=int, default=24)
    t.add_argument('--eval-every', type=int, default=50)
    t.add_argument('--state-every', type=int, default=100)
    t.add_argument('--seed', type=int, default=0)
    t.set_defaults(func=train)

    er = sub.add_parser('export-release', help='Export a Mode-P compiled-site release from any saved T2 learner-state.pt')
    er.add_argument('--state', type=Path, required=True)
    er.add_argument('--rows', type=Path, required=True)
    er.add_argument('--checkpoint', type=Path, required=True)
    er.add_argument('--base', type=Path, required=True)
    er.add_argument('--out', type=Path, required=True)
    er.add_argument('--site', default=None, help='Optional site hash to export (default: first dev-seen row).')
    er.add_argument('--device', default='cuda')
    er.add_argument('--dtype', default='bfloat16')
    er.set_defaults(func=export_release)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
