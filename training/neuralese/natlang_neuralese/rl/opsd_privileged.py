"""On-policy privileged self-distillation (OPSD) on per-call sequences of failed rollouts (`plans/NATLANG_COMPILER.md`
§8.2, Recipe T1), on the shared privileged-context mechanism.

Privileged context has one implementation: ``rl/privileged_context.py`` (custody, fact extraction with answers
withheld, the declared guidance turn and its placement) and ``rl/privileged_scoring.py`` (whole-graph twins). This module
binds it to one model call of an archived rollout, where the learner's unit is a (prompt, completion) sequence:

1. Facts: ``case_facts(case, custody=..., attempt_feedback=..., answers='withhold')``, restricted to the diagnostic
   kinds ``OPSD_FACT_KINDS`` (verifier feedback on the failed attempt, recorded reference failures, decisive evidence)
   and to what the call may see (``applicable``). Answer kinds never enter, and procedure facts that still carry an
   answer leaf are dropped by ``case_facts``.
2. Twin: the call's exact prompt token ids with the text guidance turn (``PrivilegedRendering.turn``) inserted at the
   declared place (``insertion_point``: after the leading system turn, else after BOS), followed by the student's exact
   sampled completion ids. This is the ``text-twin`` form the JOLT+ learner inserts; the runtime renders the same turn
   (``guidance_messages``; tested in ``test_jolt_privileged_context``). The twin scores exactly the student's tokens at
   exactly the student's prefix (plus the turn), checked by ``twin_sequences``.
3. Loss (``opsd_loss``): divergence between the student's distribution and the stop-gradient twin at every completion
   position, on full distributions by default (``common.stable_softmax``; ``topk`` uses top-k plus a tail bucket for
   both sides). The direction defaults to forward KL(twin || student), the JOLT+ student default (DECISIONS
   2026-11-20: its gradient does not vanish on tokens the student finds unlikely); ``'jsd'`` is the plan's symmetric
   form. The anti-shortcut mask drops positions whose sampled token the twin prefers by more than ``max_log_ratio``
   nats (outside declared syntax positions). Cross-entropy applies only where ``verified`` marks a verified
   continuation: a failed episode's own tokens are never CE targets.
4. Teacher (``train_t1 --teacher``): the twin is scored by the frozen start policy by default (``start``: the learner's
   ``reference_view``, LoRA B = 0 and the start rows). The live-policy teacher (``live``, opt-in) shares the trained
   weights, and with no CE and no anchor the objective's global minimum is any input-independent policy: the
   2026-11-26 T1 run (v2-contrast/run1) collapsed to one token (id 93800, ``))(``) at every position by step ~293,
   NLL of its own samples 25.6 nats, entropy 0.003, divergence 1e-7 (DECISIONS 2026-11-27).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .privileged_context import ANSWER_KINDS, applicable, case_facts

OPSD_SCHEMA = 'natlang.opsd-call-twin/1'
# Diagnostic facts only (the plan: verifier diff, violated clause, structural diagnosis; never the expected answer).
OPSD_FACT_KINDS = ('attempt_feedback', 'reference_failure', 'decisive')
DIVERGENCES = ('forward_kl', 'jsd', 'reverse_kl')


def opsd_facts(case, *, custody, attempt_feedback=(), invocation=None, kinds=OPSD_FACT_KINDS):
    """``(custody receipt, facts)`` the twin of one call may carry: diagnostic kinds, answers withheld, visible to it.

    ``invocation``: ``{name, role, text}`` of the call (default: the root). Raises ``CustodyError`` for a protected
    world (custody fails closed)."""
    if set(kinds) & set(ANSWER_KINDS):
        raise ValueError('OPSD twins never carry answer facts')
    receipt, facts = case_facts(case, custody=custody, attempt_feedback=attempt_feedback, answers='withhold')
    invocation = invocation or {'name': 'root', 'role': 'root', 'text': ''}
    return receipt, tuple(fact for fact in facts if fact.kind in kinds and applicable(fact, invocation))


@dataclass(frozen=True)
class CallTwin:
    """Student and twin token sequences of one call, with the completion positions aligned 1:1."""
    student_ids: tuple
    twin_ids: tuple
    student_positions: tuple   # logits rows predicting each completion token in the student sequence
    twin_positions: tuple      # the same rows in the twin sequence
    targets: tuple             # the completion token ids (the student's own samples)
    insert_at: int
    placement: str
    turn_ids: tuple
    receipt: dict


def twin_sequences(rendering, prompt_ids, completion_ids, fact_texts, *, receipt=None):
    """The call's student sequence and its privileged twin (the guidance turn inserted at the declared place)."""
    prompt, completion = list(map(int, prompt_ids)), list(map(int, completion_ids))
    if not prompt or not completion:
        raise ValueError('an OPSD twin needs a prompt and a sampled completion')
    if not fact_texts:
        raise ValueError('an OPSD twin needs at least one privileged fact')
    turn = list(rendering.turn(list(fact_texts)))
    at, placement = rendering.insertion_point(prompt)
    student = prompt + completion
    twin = prompt[:at] + turn + prompt[at:] + completion
    s_pos = tuple(range(len(prompt) - 1, len(student) - 1))
    t_pos = tuple(position + len(turn) for position in s_pos)
    if [student[p + 1] for p in s_pos] != completion or [twin[p + 1] for p in t_pos] != completion:
        raise AssertionError('twin positions do not score the student completion')
    if twin[:at] != student[:at] or twin[at + len(turn):] != student[at:]:
        raise AssertionError('the twin changed the student prefix outside the inserted turn')
    return CallTwin(tuple(student), tuple(twin), s_pos, t_pos, tuple(completion), at, placement, tuple(turn),
                    {'schema': OPSD_SCHEMA, 'rendering': rendering.digest, 'placement': placement,
                     'insert_at': at, 'turn_tokens': len(turn), **(receipt or {})})


def opsd_loss(student_logits, twin_logits, targets, *, kind='forward_kl', beta=0.5, topk=None, syntax_mask=None,
              max_log_ratio=6.0, verified=None, ce_weight=0.3):
    """Masked OPSD divergence at aligned completion positions, plus CE on verified continuations only.

    ``student_logits`` [N, V] (with gradient) and ``twin_logits`` [N, V] (detached) at the N aligned completion
    positions; ``targets`` [N] the student's sampled tokens there. ``verified`` [N] bool marks positions of verified
    continuations (a verified repair or success; default: none, so no CE). Returns ``(loss, metrics)``; the loss is the
    mean divergence over kept positions plus ``ce_weight`` times the mean CE over verified positions."""
    from ..common import stable_softmax as S
    if kind not in DIVERGENCES:
        raise ValueError(f'OPSD divergence is one of {DIVERGENCES}')
    if twin_logits.requires_grad:
        raise ValueError('the OPSD twin is a stop-gradient target')
    n = int(targets.numel())
    zero = student_logits.sum() * 0.0
    if n == 0:
        return zero, {'n': 0., 'n_kept': 0., 'shortcut_masked_frac': 0., 'divergence': 0., 'n_verified': 0., 'ce': 0.,
                      'student_nll': 0., 'student_entropy': 0.}
    targets = targets.long()
    with torch.no_grad():
        s_y = S.token_logprobs(student_logits.detach(), targets)
        s_logp = torch.log_softmax(student_logits.detach().float(), -1)
        student_entropy = float(-(s_logp.exp() * s_logp).sum(-1).mean())
        del s_logp
        t_y = S.token_logprobs(twin_logits, targets)
        log_ratio = (t_y - s_y).float()
        shortcut = log_ratio > float(max_log_ratio)
        if syntax_mask is not None:
            shortcut &= ~syntax_mask.to(torch.bool)
        kept = ~shortcut
    n_kept = int(kept.sum())
    if n_kept:
        if topk is None:
            values = S.divergence(student_logits[kept], twin_logits[kept], kind, beta=beta)
        else:
            t_logp = torch.log_softmax(twin_logits[kept].float(), -1)
            head, ids = torch.topk(t_logp, min(int(topk), t_logp.shape[-1]), -1)
            tail = torch.logsumexp(t_logp.scatter(-1, ids, float('-inf')), -1)
            values = S.topk_tail_divergence(student_logits[kept], ids, head, kind, beta=beta, teacher_log_tail=tail)
        divergence = values.float().mean()
    else:
        divergence = zero
    loss = divergence
    n_verified = 0
    ce = zero
    if verified is not None and ce_weight > 0:
        mask = verified.to(torch.bool)
        n_verified = int(mask.sum())
        if n_verified:
            ce = -S.token_logprobs(student_logits[mask], targets[mask]).float().mean()
            loss = loss + float(ce_weight) * ce
    metrics = {'n': float(n), 'n_kept': float(n_kept), 'shortcut_masked_frac': 1. - n_kept / n,
               'divergence': float(divergence.detach()), 'n_verified': float(n_verified), 'ce': float(ce.detach()),
               'mean_log_ratio': float(log_ratio[kept].mean()) if n_kept else 0.,
               # collapse telemetry: the student's NLL of its own sampled tokens and its entropy (a degenerate policy
               # shows a high NLL and a near-zero entropy while the divergence reads ~0)
               'student_nll': float(-s_y.float().mean()), 'student_entropy': student_entropy}
    return loss, metrics


# ---------------------------------------------------------------------------- Recipe T1 (OPSD tier) on real rollouts
#
# build: failed, scored rollouts of ONE policy -> per-call twins (root calls with token-only prompts), facts from the
#        group's own case rows (custody: ProtectedWorlds.load(), fails closed) and the run's verifier outcome as attempt
#        feedback; held tasks and siblings excluded by held_eval.training_exclusion_ids. Validation tasks by source-group
#        hash (the T0 rule) are kept apart for the receipt's before/after divergence.
# train: the shared LoRA learner core (frozen BF16 base, FP32 LoRA factors + AdamW, lfm3_fast.hidden); per twin a
#        no-grad forward of the current policy over the twin, a forward with gradients over the student sequence,
#        opsd_loss at the aligned completion positions. Per-step jsonl, resumable learner state, factor export,
#        and a wrong-world control: the same divergence against twins carrying another world's facts.

T1_SCHEMA = 'natlang.compiler.t1-opsd/2'   # /2: frozen start-policy teacher by default; /1 states are not resumed
TEACHERS = ('start', 'live')
# verdicts that say only that the produced value was wrong (their detail is the value itself)
ANSWER_VERDICT_OUTCOMES = frozenset({'semantic_mismatch', 'wrong_answer'})


def lfm3_rendering(tokenizer, *, consume_open=0, consume_close=0, fingerprint='0' * 64):
    """The declared guidance-turn rendering for the LFM3 ChatML template from an HF tokenizer (text form only: the
    consume markers are unused by ``turn``)."""
    from .privileged_context import PrivilegedRendering
    start, end = tokenizer.convert_tokens_to_ids(['<|im_start|>', '<|im_end|>'])
    return PrivilegedRendering('lfm3', tokenizer.bos_token_id, start, end, consume_open, consume_close, fingerprint,
                               lambda text: tokenizer(text, add_special_tokens=False)['input_ids'])


def _turn_tool_names(rollout: dict) -> list[str]:
    names = []
    for turn in rollout.get('turns') or ():
        target = turn.get('target') or {}
        for tc in target.get('tool_calls') or ():
            fn = (tc.get('function') or {}).get('name') if isinstance(tc, dict) else None
            if isinstance(fn, str) and fn:
                names.append(fn)
    return names


def synthesize_contrast_feedback(failed_rollout: dict, passing_rollout: dict) -> list[dict]:
    """Derive strictly structural, non-answer-leaking contrast diagnostics comparing a failed rollout against
    a verified passing rollout of the same task (never names return values or expected answers)."""
    task_id = failed_rollout.get('task_id')
    if not task_id or task_id != passing_rollout.get('task_id'):
        return []
    fail_tools = _turn_tool_names(failed_rollout)
    pass_tools = _turn_tool_names(passing_rollout)
    notes: list[str] = []
    omitted = sorted(set(pass_tools) - set(fail_tools) - {'return_result'})
    if omitted:
        notes.append(f"a verified run used tool(s) {', '.join(omitted)} that this attempt omitted")
    extra = sorted(set(fail_tools) - set(pass_tools) - {'return_result'})
    if extra:
        notes.append(f"this attempt invoked tool(s) {', '.join(extra)} not needed by a verified run")
    for idx, (ft, pt) in enumerate(zip(fail_tools, pass_tools)):
        if ft != pt:
            notes.append(f"tool sequence diverged at action {idx}: attempt called {ft}, whereas a verified run called {pt}")
            break
    if not notes and len(fail_tools) != len(pass_tools):
        notes.append(f"attempt took {len(fail_tools)} tool action(s) versus {len(pass_tools)} in a verified run")
    if not notes:
        return []
    return [{
        'case_id': task_id,
        'attempt': int(failed_rollout.get('sample') or 0),
        'feedback': 'Structural process diagnosis: ' + '; '.join(notes),
    }]


def build_t1_twins(collections, out, *, policy_sha256, tokenizer, max_tokens=16384, max_calls_per_episode=6,
                   max_episodes=None, held_manifest=None, use_contrast_feedback=True):
    import glob
    import json
    from pathlib import Path
    import zstandard
    from ..common.custody import ProtectedWorlds
    from ..common.jsonio import json_line, read_jsonl
    from ..train.step_verifier import SUCCESS_OUTCOMES, _completion_ids, _prompt_ids, _split_group, is_validation
    from .held_eval import training_exclusion_ids
    custody = ProtectedWorlds.load()
    rendering = lfm3_rendering(tokenizer)
    funnel = {'rollouts': 0, 'other_policy': 0, 'success_or_unscored': 0, 'held_excluded': 0, 'no_case': 0,
              'custody_refused': 0, 'no_facts': 0, 'contrast_feedback': 0,
              'episodes': 0, 'twins': 0, 'too_long': 0, 'validation_twins': 0}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + '.partial')
    with partial.open('wb') as raw, zstandard.ZstdCompressor(level=6).stream_writer(raw) as writer:
        for collection in collections:
            for group in sorted(glob.glob(str(Path(collection) / 'groups' / 'g*'))):
                from .episode_files import group_rollouts, open_lines
                rollouts_path, cases_path = group_rollouts(group), Path(group) / 'cases.jsonl'
                if rollouts_path is None or not cases_path.exists():
                    continue        # unscored (rollouts raw or compacted: episode_files)
                cases = list(read_jsonl(cases_path))
                by_id = {case['id']: case for case in cases}
                excluded = training_exclusion_ids(cases, held_manifest)
                details = {}
                rewards = Path(group) / 'rewards.jsonl'
                for row in (read_jsonl(rewards) if rewards.exists() else ()):
                    details[(row.get('task_id'), row.get('group'), row.get('sample'))] = row.get('detail')
                with open_lines(rollouts_path) as handle:
                    rows_of_group = [json.loads(line) for line in handle if line.strip()]
                passing_by_task = {}
                if use_contrast_feedback:
                    for r in rows_of_group:
                        if not r.get('unscored') and r.get('outcome') in SUCCESS_OUTCOMES and r.get('task_id'):
                            passing_by_task.setdefault(r['task_id'], r)
                for rollout in rows_of_group:
                    funnel['rollouts'] += 1
                    if (rollout.get('policy') or {}).get('checkpoint_sha256') != policy_sha256:
                        funnel['other_policy'] += 1
                        continue
                    if rollout.get('unscored') or rollout.get('outcome') in SUCCESS_OUTCOMES or not rollout.get('outcome'):
                        funnel['success_or_unscored'] += 1
                        continue
                    task_id = rollout['task_id']
                    if task_id in excluded:
                        funnel['held_excluded'] += 1
                        continue
                    case = by_id.get(task_id)
                    if case is None:
                        funnel['no_case'] += 1
                        continue
                    detail = details.get((task_id, rollout.get('group'), rollout.get('sample')))
                    feedback = [] if rollout['outcome'] in ANSWER_VERDICT_OUTCOMES else [
                        {'case_id': task_id, 'attempt': int(rollout.get('sample') or 0),
                         'feedback': f"the run ended with outcome {rollout['outcome']}"
                                     + (f'; verifier detail: {detail}' if detail else '')}]
                    if feedback:
                        funnel['structural_feedback'] = funnel.get('structural_feedback', 0) + 1
                    if use_contrast_feedback and task_id in passing_by_task:
                        contrast = synthesize_contrast_feedback(rollout, passing_by_task[task_id])
                        if contrast:
                            feedback.extend(contrast)
                            funnel['contrast_feedback'] += 1
                    try:
                        receipt, facts = opsd_facts(case, custody=custody, attempt_feedback=feedback)
                    except Exception:
                        funnel['custody_refused'] += 1
                        continue
                    if not facts:
                        funnel['no_facts'] += 1
                        continue
                    texts = [fact.text for fact in facts]
                    roots = [index for index, turn in enumerate(rollout['turns'])
                             if turn.get('invocation_role') == 'root' and _prompt_ids(turn) is not None
                             and _completion_ids(turn)]
                    if len(roots) > max_calls_per_episode:   # evenly spread over the episode
                        step = len(roots) / max_calls_per_episode
                        roots = [roots[int(i * step)] for i in range(max_calls_per_episode)]
                    validation = is_validation(rollout)
                    kept = 0
                    for index in roots:
                        turn = rollout['turns'][index]
                        twin = twin_sequences(rendering, _prompt_ids(turn), _completion_ids(turn), texts,
                                              receipt={'custody': receipt, 'facts': [f.id for f in facts]})
                        if len(twin.twin_ids) > max_tokens:
                            funnel['too_long'] += 1
                            continue
                        kept += 1
                        funnel['twins'] += 1
                        funnel['validation_twins'] += int(validation)
                        writer.write(json_line({'task_id': task_id, 'family': rollout.get('family'),
                                                'split_group': _split_group(rollout), 'validation': validation,
                                                'episode': f"{Path(group).name}/{rollout['group']}/{rollout['sample']}",
                                                'turn': index, 'outcome': rollout['outcome'], 'facts': texts,
                                                'prompt_ids': list(twin.student_ids[:len(twin.student_ids) - len(twin.targets)]),
                                                'completion_ids': list(twin.targets)}).encode('utf-8') + b'\n')
                    funnel['episodes'] += int(kept > 0)
                    if max_episodes and funnel['episodes'] >= max_episodes:
                        break
                if max_episodes and funnel['episodes'] >= max_episodes:
                    break
            if max_episodes and funnel['episodes'] >= max_episodes:
                break
    partial.replace(out)
    return funnel


def _twin_rows(path):
    import io
    import json
    import zstandard
    with open(path, 'rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        for line in io.TextIOWrapper(reader, encoding='utf-8'):
            if line.strip():
                yield json.loads(line)


ACTIVATION_GIB = 4.0   # saved activations of a student forward; layers beyond this recompute (lfm3_fast.set_checkpointing)


def _positions_hidden(learner, ids, positions, *, grad, start=False):
    """Final hidden rows at ``positions``: the live policy, or with ``start`` the frozen start policy (no grad)."""
    from ..train import lfm3_fast
    tensor = torch.tensor(ids, device=learner.device)
    if start:
        if grad:
            raise ValueError('the start policy is a frozen teacher')
        with learner.reference_view(), torch.no_grad():   # bare base adapters and the start rows
            h = lfm3_fast.hidden(learner.model, tensor)
        return h[torch.tensor(positions, device=learner.device)]
    if grad:
        # the learner core's recompute plan: a 16k-token student forward kept every layer's activations and OOMed the L4
        lfm3_fast.set_checkpointing(learner.model, len(ids), ACTIVATION_GIB, parameterization='lora+rows')
    with torch.set_grad_enabled(grad):
        h = lfm3_fast.hidden(learner.model, tensor, adapters=learner.bundles, trainable_rows=learner.trainable_rows)
        return h[torch.tensor(positions, device=learner.device)]


def _head(learner, rows, *, start=False):
    """FP32 logits of ``rows`` under the policy's head, the trainable Neuralese rows scoring their ids (the rule of
    ``lfm3_fast._head_logits_chunk``). Differentiable: lfm3_fast's FP32-output ``torch.mm(out_dtype=...)`` has no
    autograd formula (``derivative for aten::mm is not implemented`` in the first T1 backward), so the head runs as an
    FP32 matmul against a cached FP32 copy of the frozen BF16 head weight."""
    weight = learner.model.lm_head.weight
    cached = getattr(learner, '_opsd_head_fp32', None)
    if cached is None or cached.device != weight.device:
        cached = weight.detach().float()
        learner._opsd_head_fp32 = cached
    logits = rows.float() @ cached.t()
    special = None if start else learner.trainable_rows   # start rows score through the frozen head
    if special is not None and special.token_ids.numel():
        logits = logits.clone()
        logits[:, special.token_ids] = rows.float() @ special.rows.float().t()
    return logits


def twin_loss(learner, twin, *, grad, scale=1.0, chunk=256, teacher='start', **loss_options):
    """``opsd_loss`` of one twin, its full-vocabulary logits formed ``chunk`` completion positions at a time (a long
    completion's [n, V] FP32 logits do not fit beside the model). With ``grad`` the chunks backpropagate one by one
    through the shared student hidden states, each weighted by its share of the positions times ``scale``. Returns
    the position-weighted metrics. ``teacher``: ``'start'`` (frozen start policy, default) or ``'live'``."""
    if teacher not in TEACHERS:
        raise ValueError(f'OPSD teacher is one of {TEACHERS}')
    start_teacher = teacher == 'start'
    with torch.no_grad():
        t_rows = (_positions_hidden(learner, twin.twin_ids, twin.twin_positions, grad=False, start=True) if start_teacher
                  else _positions_hidden(learner, twin.twin_ids, twin.twin_positions, grad=False))
    s_rows = _positions_hidden(learner, twin.student_ids, twin.student_positions, grad=grad)
    targets = torch.tensor(twin.targets, device=s_rows.device)
    n = len(twin.targets)
    totals = {'divergence': 0., 'shortcut_masked_frac': 0., 'student_nll': 0., 'student_entropy': 0., 'n': float(n)}
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        share = (stop - start) / n
        with torch.no_grad():
            t = _head(learner, t_rows[start:stop], start=True) if start_teacher else _head(learner, t_rows[start:stop])
        with torch.set_grad_enabled(grad):
            s = _head(learner, s_rows[start:stop])
            loss, metrics = opsd_loss(s, t, targets[start:stop], **loss_options)
            if grad:
                (loss * share * scale).backward(retain_graph=stop < n)
        for key in ('divergence', 'shortcut_masked_frac', 'student_nll', 'student_entropy'):
            totals[key] += metrics[key] * share
    return totals


def train_t1(twins_path, out, *, checkpoint, base, tokenizer, rank=128, alpha=128.0, lr=3e-5, warmup=20,
             batch_twins=8, max_steps=None, kind='forward_kl', topk=None, max_log_ratio=6.0, state_every=25,
             validation_limit=64, seed=0, teacher='start'):
    import json
    import random
    from pathlib import Path
    from ..common.code_provenance import code_provenance
    from ..train.lfm3_learner import AdapterSpec, Lfm3Learner
    from ..train.learner_state import load_learner_state, save_learner_state
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rendering = lfm3_rendering(tokenizer)
    rows = list(_twin_rows(twins_path))
    train_rows = [row for row in rows if not row['validation']]
    val_rows = [row for row in rows if row['validation']][:validation_limit]
    rng = random.Random(seed)
    rng.shuffle(train_rows)
    learner = Lfm3Learner.from_checkpoint(checkpoint, base, AdapterSpec.for_sft(rank=rank, alpha=alpha), seed=seed,
                                          compile_layers=False)
    optimizer = learner.build_optimizer(lr)
    step = 0
    state_dir = out / 'learner-state'
    if (state_dir / 'learner_manifest.json').exists():
        saved = json.loads((state_dir / 'learner_manifest.json').read_text()).get('objective_version')
        if saved != T1_SCHEMA:
            raise ValueError(f'{state_dir} holds objective {saved!r}, not {T1_SCHEMA!r}: start a new run directory')
        restored = load_learner_state(state_dir, learner, optimizer)
        step = int(restored.get('learner_step') or 0)
    total_steps = max_steps or max(1, len(train_rows) // batch_twins)

    def twin_of(row, facts=None):
        return twin_sequences(rendering, row['prompt_ids'], row['completion_ids'], facts or row['facts'])

    def divergence_on(sample, wrong=False):
        values = []
        for i, row in enumerate(sample):
            facts = sample[(i + 1) % len(sample)]['facts'] if wrong and len(sample) > 1 else None
            twin = twin_of(row, facts)
            with torch.no_grad():
                metrics = twin_loss(learner, twin, grad=False, kind=kind, topk=topk, max_log_ratio=max_log_ratio,
                                    teacher=teacher)
            values.append(metrics['divergence'])
        return sum(values) / max(1, len(values))

    receipt = {'schema': T1_SCHEMA, 'twins': str(twins_path), 'checkpoint': str(checkpoint), 'rows': len(rows),
               'train_twins': len(train_rows), 'validation_twins': len(val_rows), 'rank': rank, 'alpha': alpha,
               'lr': lr, 'kind': kind, 'topk': topk, 'teacher': teacher, 'max_log_ratio': max_log_ratio, 'code': code_provenance()}
    if step == 0 and val_rows:
        receipt['validation_before'] = {'divergence': divergence_on(val_rows),
                                        'wrong_world_divergence': divergence_on(val_rows, wrong=True)}
    log = (out / 'train.jsonl').open('a')
    cursor = step * batch_twins
    while step < total_steps and cursor < len(train_rows):
        batch = train_rows[cursor:cursor + batch_twins]
        cursor += batch_twins
        scale = min(1.0, (step + 1) / max(1, warmup))
        for group in optimizer.param_groups:
            group['lr'] = lr * scale * group.get('lr_scale', 1.0)
        optimizer.zero_grad(set_to_none=True)
        sums = {'divergence': 0., 'shortcut_masked_frac': 0., 'student_nll': 0., 'student_entropy': 0., 'n': 0.}
        for row in batch:
            twin = twin_of(row)
            metrics = twin_loss(learner, twin, grad=True, scale=1.0 / len(batch), kind=kind, topk=topk,
                                max_log_ratio=max_log_ratio, teacher=teacher)
            for key in sums:
                sums[key] += metrics[key] / len(batch)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(learner.trainable_parameters(), 1.0))
        optimizer.step()
        step += 1
        log.write(json.dumps({'step': step, 'lr': lr * scale, 'grad_norm': grad_norm, **sums}) + '\n')
        log.flush()
        if step % state_every == 0:
            save_learner_state(state_dir, learner, optimizer, learner_step=step, source_checkpoint_id=str(checkpoint),
                               objective_version=T1_SCHEMA)
    log.close()
    save_learner_state(state_dir, learner, optimizer, learner_step=step, source_checkpoint_id=str(checkpoint),
                       objective_version=T1_SCHEMA)
    if val_rows:
        receipt['validation_after'] = {'divergence': divergence_on(val_rows),
                                       'wrong_world_divergence': divergence_on(val_rows, wrong=True)}
    tensors, scales = learner.factor_tensors()
    torch.save({'tensors': tensors, 'scales': scales, 'spec': learner.spec.to_dict()}, out / 'factors.pt')
    receipt['steps'] = step
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main(argv=None):
    import argparse
    import json
    from pathlib import Path
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description='Recipe T1: OPSD per-call twins on real failed rollouts')
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--collection', action='append', required=True)
    b.add_argument('--policy-sha256', required=True)
    b.add_argument('--tokenizer', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--max-episodes', type=int)
    b.add_argument('--held-manifest')
    b.add_argument('--contrast-feedback', action=argparse.BooleanOptionalAction, default=True,
                   help='Synthesize structural tool-sequence contrast feedback for failed episodes without verifier notes (default: true).')
    t = sub.add_parser('train')
    t.add_argument('--twins', required=True)
    t.add_argument('--checkpoint', required=True)
    t.add_argument('--base', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--max-steps', type=int)
    t.add_argument('--batch-twins', type=int, default=8)
    t.add_argument('--lr', type=float, default=3e-5)
    t.add_argument('--kind', choices=DIVERGENCES, default='forward_kl')
    t.add_argument('--topk', type=int)
    t.add_argument('--state-every', type=int, default=25, help='learner-state save interval in steps')
    t.add_argument('--teacher', choices=TEACHERS, default='start',
                   help='twin scorer: the frozen start policy (default) or the live policy (opt-in; collapsed T1 run1)')
    args = parser.parse_args(argv)
    if args.command == 'build':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        result = build_t1_twins(args.collection, args.out, policy_sha256=args.policy_sha256, tokenizer=tokenizer,
                                max_episodes=args.max_episodes, held_manifest=args.held_manifest,
                                use_contrast_feedback=args.contrast_feedback)
        from ..common.code_provenance import code_provenance
        Path(args.out + '.json').write_text(json.dumps({'schema': T1_SCHEMA, 'stage': 'build', 'funnel': result,
                                                        'collections': args.collection,
                                                        'policy_sha256': args.policy_sha256,
                                                        'code': code_provenance()}, indent=2) + '\n')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        result = train_t1(args.twins, args.out, checkpoint=args.checkpoint, base=args.base, tokenizer=tokenizer,
                          max_steps=args.max_steps, batch_twins=args.batch_twins, lr=args.lr, kind=args.kind,
                          topk=args.topk, state_every=args.state_every, teacher=args.teacher)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
