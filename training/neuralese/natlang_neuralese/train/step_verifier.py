"""Hidden-State Process Step-Verifier Head & Prefix-Shared Branch Scorer (Recipe T0, §6.2 & §9.2).

Attaches a lightweight 2-layer FP32 MLP (~2M params at hidden_dim=2048) to LFM3-1.7B's final
hidden state at tool-call and `return_result` boundaries. Trained on existing pass/fail
student rollout archives to score candidate branches in prefix-shared `branch@4` search.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F


class StepVerifierHead(nn.Module):
    """Predicts [episode_success_logit, immediate_error_logit] from LFM3 final hidden state."""

    def __init__(self, hidden_dim: int, mlp_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        if hidden_dim <= 0 or mlp_dim <= 0:
            raise ValueError("hidden_dim and mlp_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.mlp_dim = int(mlp_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 2),
        )
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [..., hidden_dim] -> logits [..., 2] in FP32."""
        return self.net(self.norm(hidden.float()))

    @staticmethod
    def rich_features(hidden: torch.Tensor, prompt_end: int, boundary: int,
                      scalars: list[float] | tuple[float, ...] | None = None) -> torch.Tensor:
        """Build the exact ``feature_mode='rich'`` vector ``[h_last, h_last - h_prompt_end, h_mean_completion, scalars]``
        from a sequence hidden-state tensor ``[T, H]`` so ``branch@N`` inference matches T0 training 1:1."""
        h = hidden.float()
        b = max(0, min(int(boundary), h.shape[0] - 1))
        pe = max(0, min(int(prompt_end), b))
        h_last = h[b]
        h_delta = h_last - h[pe]
        start_c = min(pe + 1, b)
        h_mean = h[start_c:b + 1].mean(dim=0)
        sc = torch.tensor(list(scalars) if scalars is not None else [0.0] * 5, device=h.device, dtype=torch.float32)
        return torch.cat([h_last, h_delta, h_mean, sc], dim=-1)

    @torch.no_grad()
    def success_probability(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.forward(hidden)
        return torch.sigmoid(logits[..., 0])

    @torch.no_grad()
    def score_boundary(self, sequence_hidden: torch.Tensor, prompt_end: int, boundary: int,
                       scalars: list[float] | tuple[float, ...] | None = None) -> float:
        """Score a candidate branch boundary; automatically uses rich features when head was trained on rich features."""
        h = sequence_hidden.float()
        base_dim = int(h.shape[-1])
        if self.hidden_dim == 3 * base_dim + 5:
            feat = self.rich_features(h, prompt_end, boundary, scalars).unsqueeze(0)
        else:
            b = max(0, min(int(boundary), h.shape[0] - 1))
            feat = h[b:b + 1]
        return float(self.success_probability(feat.to(next(self.parameters()).device)).item())


def compute_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Exact Mann-Whitney / Wilcoxon rank AUROC with tie handling (no sklearn dependency)."""
    s = scores.detach().float().flatten()
    y = labels.detach().float().flatten()
    pos = y > 0.5
    n_pos = int(pos.sum().item())
    n_neg = int((~pos).sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    s_sorted = s[order]
    y_sorted = pos[order]
    # Assign average ranks for ties (1-indexed)
    n = s_sorted.numel()
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and float(s_sorted[j].item()) == float(s_sorted[i].item()):
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_rank
        i = j
    pos_rank_sum = float(ranks[y_sorted].sum().item())
    return (pos_rank_sum - 0.5 * n_pos * (n_pos + 1)) / float(n_pos * n_neg)


@dataclass(frozen=True)
class BranchCandidate:
    branch_id: str
    ensures_ok: bool
    type_ok: bool
    tool_exit_ok: bool
    verifier_score: float
    mean_logprob: float = 0.0


def rank_branches(candidates: list[BranchCandidate]) -> list[BranchCandidate]:
    """Rank prefix-shared branch@N candidates: crisp checks strictly first, then Step-Verifier V(h_t)."""
    return sorted(
        candidates,
        key=lambda c: (
            1 if (c.ensures_ok and c.type_ok and c.tool_exit_ok) else 0,
            1 if c.type_ok else 0,
            1 if c.tool_exit_ok else 0,
            float(c.verifier_score),
            float(c.mean_logprob),
        ),
        reverse=True,
    )


@dataclass(frozen=True)
class VerifierTrainConfig:
    epochs: int = 8
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.05
    error_loss_weight: float = 0.3
    clip_norm: float = 1.0
    dropout: float = 0.2
    pairwise_weight: float = 0.25


def train_step_verifier(
    head: StepVerifierHead,
    *,
    train_hidden: torch.Tensor,
    train_success: torch.Tensor,
    train_error: torch.Tensor | None = None,
    val_hidden: torch.Tensor | None = None,
    val_success: torch.Tensor | None = None,
    cfg: VerifierTrainConfig = VerifierTrainConfig(),
) -> dict[str, float]:
    """Train StepVerifierHead in FP32 with cosine LR schedule and return final metrics."""
    device = next(head.parameters()).device
    n = int(train_hidden.shape[0])
    if n == 0:
        raise ValueError("train_hidden is empty")
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, math.ceil(n / cfg.batch_size))
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.lr * 0.1)

    head.train()
    last_loss = 0.0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb = train_hidden[idx].to(device=device, dtype=torch.float32)
            yb = train_success[idx].to(device=device, dtype=torch.float32)
            logits = head(xb)
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], yb)
            if train_error is not None and cfg.error_loss_weight > 0.0:
                eb = train_error[idx].to(device=device, dtype=torch.float32)
                loss = loss + cfg.error_loss_weight * F.binary_cross_entropy_with_logits(logits[:, 1], eb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_norm)
            optimizer.step()
            scheduler.step()
            last_loss = float(loss.detach().item())

    head.eval()
    with torch.no_grad():
        tr_probs = head.success_probability(train_hidden.to(device=device, dtype=torch.float32)).cpu()
        tr_auroc = compute_auroc(tr_probs, train_success.cpu())
        val_auroc = float("nan")
        if val_hidden is not None and val_success is not None and len(val_hidden) > 0:
            v_probs = head.success_probability(val_hidden.to(device=device, dtype=torch.float32)).cpu()
            val_auroc = compute_auroc(v_probs, val_success.cpu())
    return {"train_loss": last_loss, "train_auroc": tr_auroc, "val_auroc": val_auroc}


# ---------------------------------------------------------------------------- Recipe T0 on archived rollouts
#
# build:    archived on-policy rollouts (natlang.rollout/1) of ONE policy -> per-call token sequences with the positions
#           where each model call's sampled completion ends (a tool call or return_result), labelled with the episode's
#           outcome. Turns of one invocation whose prompts extend the previous turn's prompt + completion share one
#           sequence (one forward reads every boundary). Held tasks and their siblings are excluded through
#           held_eval.training_exclusion_ids over the groups' own case rows.
# features: the policy that sampled them (exact checkpoint sha) reads each sequence; the final-norm hidden state at
#           every boundary is stored (BF16) with its labels.
# train:    StepVerifierHead in FP32 on a task-disjoint split (validation tasks by hash of their split group), pass/fail
#           balanced per family by sample weights; per-epoch receipts, validation AUROC overall and per family, and the
#           label-permuted control AUROC (must sit near 0.5; a control far from it means leakage or a bug).

T0_SCHEMA = 'natlang.compiler.t0-step-verifier/1'
SUCCESS_OUTCOMES = frozenset({'task_success', 'success'})


def _prompt_ids(turn):
    parts = (turn.get('sampling') or {}).get('prompt') or []
    if not parts or any(part.get('kind') != 'tokens' for part in parts):
        return None
    return [int(token) for part in parts for token in part['token_ids']]


def _completion_ids(turn):
    return [int(sample['token_id']) for sample in (turn.get('sampling') or {}).get('samples') or []]


def _turn_scalars(turn, *, turn_idx: int, total_turns: int, completion_len: int) -> list[float]:
    """Compact turn-level scalar telemetry (FP32) helping distinguish failing loops/low-confidence turns from passes."""
    samples = (turn.get('sampling') or {}).get('samples') or []
    logps = [float(s['logprob']) for s in samples if isinstance(s, dict) and s.get('logprob') is not None]
    mean_lp = (sum(logps) / len(logps)) if logps else 0.0
    min_lp = min(logps) if logps else 0.0
    target = turn.get('target') or {}
    tcs = target.get('tool_calls') or []
    is_return = 1.0 if any(((tc.get('function') or {}).get('name') == 'return_result') for tc in tcs if isinstance(tc, dict)) else 0.0
    progress = float(turn_idx) / float(max(1, total_turns - 1))
    return [progress, math.log1p(max(0, completion_len)) / 8.0, max(-10.0, min(0.0, mean_lp)) / 5.0,
            max(-20.0, min(0.0, min_lp)) / 10.0, is_return]


def rollout_sequences(rollout, *, max_tokens=32768, include_rich=False):
    """Per-invocation sequences of one rollout: ``[{ids, boundaries, turns}]`` (plus ``prompt_ends`` and ``scalars``
    when ``include_rich=True`` for rich delta/telemetry extraction).

    ``boundaries`` are positions (indices into ``ids``) of each call's last sampled token: the hidden state there has
    read the whole call. A turn whose prompt does not extend its predecessor's sequence starts a new sequence. Turns
    with non-token prompt parts (imported native states) are skipped."""
    chains = {}
    all_turns = rollout.get('turns') or ()
    total_turns = len(all_turns)
    for index, turn in enumerate(all_turns):
        prompt, completion = _prompt_ids(turn), _completion_ids(turn)
        if prompt is None or not completion:
            continue
        prompt_end = len(prompt) - 1
        scalars = _turn_scalars(turn, turn_idx=index, total_turns=total_turns, completion_len=len(completion)) if include_rich else None
        key = turn.get('invocation_id') or f'turn-{index}'
        sequences = chains.setdefault(key, [])
        current = sequences[-1] if sequences else None
        if current is not None and prompt[:len(current['ids'])] == current['ids']:
            ids = prompt + completion
            if len(ids) <= max_tokens:
                current['ids'] = ids
                current['boundaries'].append(len(ids) - 1)
                current['turns'].append(index)
                if include_rich:
                    current['prompt_ends'].append(prompt_end)
                    current['scalars'].append(scalars)
                continue
        ids = prompt + completion
        if len(ids) > max_tokens:
            continue
        entry = {'ids': ids, 'boundaries': [len(ids) - 1], 'turns': [index]}
        if include_rich:
            entry['prompt_ends'] = [prompt_end]
            entry['scalars'] = [scalars]
        sequences.append(entry)
    return [sequence for sequences in chains.values() for sequence in sequences]


def _split_group(rollout):
    groups = rollout.get('source_groups') or ()
    return str(sorted(groups)[0] if groups else rollout['task_id'])


def is_validation(rollout, fraction=0.1):
    """Task-disjoint validation membership by a hash of the rollout's first source group (siblings stay together)."""
    import hashlib
    digest = hashlib.sha256(_split_group(rollout).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64 < float(fraction)


def build_t0_sequences(collections, out, *, policy_sha256, max_tokens=32768, max_rollouts=None, held_manifest=None):
    """Write ``out`` (jsonl.zst) of per-call sequences from the rollouts under ``collections`` matching
    ``policy_sha256`` (a single sha, an iterable of shas, or ``'any'`` to pool multiple nearby policies on the same
    backbone); returns the funnel."""
    import glob
    import json
    from pathlib import Path
    from ..rl.held_eval import training_exclusion_ids
    from ..common.jsonio import json_line, read_jsonl
    import zstandard
    if isinstance(policy_sha256, str):
        allow_any = policy_sha256.lower() == 'any'
        allowed_policies = None if allow_any else frozenset({policy_sha256})
    else:
        items = list(policy_sha256)
        allow_any = any(str(x).lower() == 'any' for x in items)
        allowed_policies = None if allow_any else frozenset(items)
    funnel = {'groups': 0, 'rollouts': 0, 'other_policy': 0, 'unscored': 0, 'held_excluded': 0, 'kept_rollouts': 0,
              'sequences': 0, 'boundaries': 0, 'success_rollouts': 0, 'validation_rollouts': 0}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + '.partial')
    with partial.open('wb') as raw, zstandard.ZstdCompressor(level=6).stream_writer(raw) as writer:
        for collection in collections:
            for group in sorted(glob.glob(str(Path(collection) / 'groups' / 'g*'))):
                from ..rl.episode_files import group_rollouts, open_lines
                rollouts_path, cases_path = group_rollouts(group), Path(group) / 'cases.jsonl'
                if rollouts_path is None or not cases_path.exists():
                    continue        # unscored (rollouts raw or compacted: episode_files)
                funnel['groups'] += 1
                cases = list(read_jsonl(cases_path))
                excluded = training_exclusion_ids(cases, held_manifest)
                with open_lines(rollouts_path) as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        rollout = json.loads(line)
                        funnel['rollouts'] += 1
                        ckpt_sha = (rollout.get('policy') or {}).get('checkpoint_sha256')
                        if allowed_policies is not None and ckpt_sha not in allowed_policies:
                            funnel['other_policy'] += 1
                            continue
                        if rollout.get('unscored') or not rollout.get('outcome'):
                            funnel['unscored'] += 1
                            continue
                        if rollout['task_id'] in excluded:
                            funnel['held_excluded'] += 1
                            continue
                        success = rollout['outcome'] in SUCCESS_OUTCOMES
                        validation = is_validation(rollout)
                        sequences = rollout_sequences(rollout, max_tokens=max_tokens, include_rich=True)
                        if not sequences:
                            continue
                        funnel['kept_rollouts'] += 1
                        funnel['success_rollouts'] += int(success)
                        funnel['validation_rollouts'] += int(validation)
                        episode = f"{Path(group).parent.parent.name}/{Path(group).name}/{rollout['group']}/{rollout['sample']}"
                        for number, sequence in enumerate(sequences):
                            funnel['sequences'] += 1
                            funnel['boundaries'] += len(sequence['boundaries'])
                            writer.write(json_line({'episode': episode, 'sequence': number,
                                                    'task_id': rollout['task_id'], 'family': rollout.get('family'),
                                                    'split_group': _split_group(rollout), 'outcome': rollout['outcome'],
                                                    'success': success, 'validation': validation,
                                                    **sequence}).encode('utf-8') + b'\n')
                        if max_rollouts and funnel['kept_rollouts'] >= max_rollouts:
                            break
                if max_rollouts and funnel['kept_rollouts'] >= max_rollouts:
                    break
            if max_rollouts and funnel['kept_rollouts'] >= max_rollouts:
                break
    partial.replace(out)
    return funnel


def _sequence_rows(path):
    import io
    import json
    import zstandard
    with open(path, 'rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        for line in io.TextIOWrapper(reader, encoding='utf-8'):
            if line.strip():
                yield json.loads(line)


@torch.no_grad()
def extract_t0_features(sequences_path, out, *, checkpoint, base, checkpoint_sha256, device='cuda', shard_rows=4096,
                        resume=True, feature_mode='rich'):
    """Boundary features of the sampling policy; resumable shards ``out/shard-NNNNN.pt``.

    ``feature_mode='rich'`` (default): concatenates ``[h_last, h_last - h_prompt_end, h_mean_completion, scalars]``
    so the head sees the turn-specific delta (canceling static prompt/task difficulty), completion summary, and
    logprob/progress telemetry. ``feature_mode='last'`` keeps the legacy single-token ``h_last`` vector."""
    import hashlib
    import json
    from pathlib import Path
    from .lfm3_learner import load_base_model
    if feature_mode not in ('rich', 'last'):
        raise ValueError(f"feature_mode must be 'rich' or 'last', got {feature_mode!r}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    with open(checkpoint, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            sha.update(block)
    if sha.hexdigest() != checkpoint_sha256:
        raise ValueError(f'checkpoint sha {sha.hexdigest()} is not the sampling policy {checkpoint_sha256}')
    model, *_ = load_base_model(checkpoint, base, device=device, dtype=torch.bfloat16)
    model.eval()
    done = {p.name for p in out.glob('shard-*.pt')} if resume else set()
    hidden, meta, shard, rows_seen = [], [], 0, 0

    def flush():
        nonlocal hidden, meta, shard
        if not hidden:
            return
        name = f'shard-{shard:05d}.pt'
        if name not in done:
            tmp = out / (name + '.partial')
            torch.save({'hidden': torch.cat(hidden).to(torch.bfloat16).cpu(), 'meta': meta}, tmp)
            tmp.replace(out / name)
        hidden, meta, shard = [], [], shard + 1

    for row in _sequence_rows(sequences_path):
        rows_seen += 1
        if f'shard-{shard:05d}.pt' not in done:
            ids = torch.tensor([row['ids']], device=device)
            h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0].float()
            bounds = row['boundaries']
            if feature_mode == 'rich' and row.get('prompt_ends'):
                p_ends = row['prompt_ends']
                sc_list = row.get('scalars') or [[0.0] * 5 for _ in bounds]
                feats = []
                for b, pe, sc in zip(bounds, p_ends, sc_list):
                    pe_clamped = max(0, min(int(pe), int(b)))
                    h_last = h[b]
                    h_delta = h_last - h[pe_clamped]
                    start_c = min(pe_clamped + 1, int(b))
                    h_mean = h[start_c:int(b) + 1].mean(dim=0)
                    sc_t = torch.tensor(sc, device=h.device, dtype=torch.float32)
                    feats.append(torch.cat([h_last, h_delta, h_mean, sc_t], dim=-1))
                hidden.append(torch.stack(feats, dim=0))
            else:
                hidden.append(h[torch.tensor(bounds, device=device)])
        meta.extend({'episode': row['episode'], 'sequence': row['sequence'], 'boundary': b, 'turn': t,
                     'task_id': row['task_id'], 'family': row['family'], 'split_group': row['split_group'],
                     'success': row['success'], 'validation': row['validation']}
                    for b, t in zip(row['boundaries'], row['turns']))
        if len(meta) >= shard_rows:
            if f'shard-{shard:05d}.pt' in done:
                hidden, meta, shard = [], [], shard + 1
            else:
                flush()
    if f'shard-{shard:05d}.pt' in done:
        hidden, meta = [], []
    flush()
    readout_desc = ('[h_last, h_last - h_prompt_end, h_mean_completion, scalars]'
                    if feature_mode == 'rich' else 'final-norm hidden state at the last sampled token of each call')
    receipt = {'schema': T0_SCHEMA, 'stage': 'features', 'sequences': str(sequences_path), 'rows': rows_seen,
               'checkpoint_sha256': checkpoint_sha256, 'base': str(base), 'dtype': 'bfloat16',
               'feature_mode': feature_mode, 'readout': readout_desc}
    (out / 'features.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def _per_family_auroc(scores, labels, families, minimum=10):
    out = {}
    for family in sorted(set(families)):
        idx = torch.tensor([i for i, f in enumerate(families) if f == family])
        y = labels[idx]
        if int(y.sum()) >= minimum and int((1 - y).sum()) >= minimum:
            out[family] = compute_auroc(scores[idx], y)
    return out


def within_task_auroc(scores: torch.Tensor, labels: torch.Tensor, task_ids: list[str]) -> tuple[float, int]:
    """Mean per-task AUROC over tasks with both passing and failing boundaries — the exact metric branch@N needs."""
    by_task: dict[str, list[int]] = {}
    for i, tid in enumerate(task_ids):
        by_task.setdefault(str(tid), []).append(i)
    aurocs = []
    for members in by_task.values():
        idx = torch.tensor(members, dtype=torch.long)
        y_t = labels[idx]
        if float(y_t.sum()) >= 1.0 and float((1.0 - y_t).sum()) >= 1.0:
            a = compute_auroc(scores[idx], y_t)
            if not math.isnan(a):
                aurocs.append(a)
    return (sum(aurocs) / len(aurocs) if aurocs else float('nan')), len(aurocs)


def pairwise_ranking_loss(logits_1d: torch.Tensor, labels: torch.Tensor, task_ids: list[str]) -> torch.Tensor:
    """Within-task RankNet logistic loss over all (pass, fail) pairs sharing ``task_id`` in the batch."""
    by_task: dict[str, list[int]] = {}
    for i, tid in enumerate(task_ids):
        by_task.setdefault(str(tid), []).append(i)
    terms = []
    for members in by_task.values():
        pos = [i for i in members if float(labels[i]) > 0.5]
        neg = [i for i in members if float(labels[i]) <= 0.5]
        if not pos or not neg:
            continue
        p_s = logits_1d[pos][:, None]
        n_s = logits_1d[neg][None, :]
        terms.append(F.softplus(-(p_s - n_s)).mean())
    if not terms:
        return logits_1d.sum() * 0.0
    return torch.stack(terms).mean()


def train_t0(features_dir, out, *, cfg: VerifierTrainConfig = VerifierTrainConfig(), seed=0, device='cpu',
             progress_weighting: bool = True):
    """Train the head on task-disjoint splits of stored features; write head, per-epoch jsonl and receipt.

    ``progress_weighting`` (default True) up-weights late/decisive boundaries within each episode
    (``0.25 + 0.75 * progress``). When ``cfg.pairwise_weight > 0``, batches are grouped by task so within-task
    (pass, fail) pairs train a RankNet term that cancels static task-difficulty confounding."""
    import json
    import random
    from pathlib import Path
    from ..common.code_provenance import code_provenance
    torch.manual_seed(seed)
    rng = random.Random(seed)
    features_dir, out = Path(features_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    hidden, meta = [], []
    for shard in sorted(features_dir.glob('shard-*.pt')):
        data = torch.load(shard, map_location='cpu')
        hidden.append(data['hidden'])
        meta.extend(data['meta'])
    x = torch.cat(hidden).float()
    y = torch.tensor([float(m['success']) for m in meta])
    val = torch.tensor([bool(m['validation']) for m in meta])
    fam = [m['family'] or 'unknown' for m in meta]
    tids = [str(m.get('task_id') or m.get('episode') or i) for i, m in enumerate(meta)]
    tr_idx, va_idx = (~val).nonzero().squeeze(1), val.nonzero().squeeze(1)
    # pass/fail balance per family (and equal family mass) by sample weights, optionally scaled by episode progress
    progress_scale = torch.ones(len(meta))
    if progress_weighting:
        by_ep: dict[str, list[int]] = {}
        for i, m in enumerate(meta):
            by_ep.setdefault(str(m.get('episode') or i), []).append(i)
        for members in by_ep.values():
            members.sort(key=lambda idx: (meta[idx].get('turn', 0), meta[idx].get('boundary', 0)))
            denom = max(1, len(members) - 1)
            for rank, idx in enumerate(members):
                progress_scale[idx] = 0.25 + 0.75 * (rank / denom)
    weights = torch.zeros(len(meta))
    for family in set(fam[i] for i in tr_idx.tolist()):
        members = [i for i in tr_idx.tolist() if fam[i] == family]
        for label in (0., 1.):
            cls = [i for i in members if y[i] == label]
            if cls:
                cls_scale = progress_scale[cls]
                denom_scale = float(cls_scale.sum())
                weights[cls] = cls_scale / (2.0 * denom_scale if denom_scale > 0 else 2.0 * len(cls))
    head = StepVerifierHead(x.shape[-1], dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = len(tr_idx)
    steps = max(1, cfg.epochs * math.ceil(n / cfg.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=cfg.lr * 0.1)
    log = (out / 'train.jsonl').open('w')
    sampler_weights = weights[tr_idx]
    # Task-grouped index for within-task pairwise ranking batches
    tr_by_task: dict[str, list[int]] = {}
    for i in tr_idx.tolist():
        tr_by_task.setdefault(tids[i], []).append(i)
    task_keys = list(tr_by_task.keys())
    task_weights = torch.tensor([float(weights[tr_by_task[tk]].sum()) for tk in task_keys], dtype=torch.float32)
    mixed_tasks = [t for t, ms in tr_by_task.items()
                   if any(float(y[i]) > 0.5 for i in ms) and any(float(y[i]) <= 0.5 for i in ms)]
    va_tids = [tids[i] for i in va_idx.tolist()]
    for epoch in range(cfg.epochs):
        head.train()
        if cfg.pairwise_weight > 0.0 and mixed_tasks and float(task_weights.sum()) > 0:
            # Family-balanced task-grouped sampling: preserves equal family mass while guaranteeing same-task pairs
            order_list: list[int] = []
            while len(order_list) < n:
                drawn_tasks = torch.multinomial(task_weights, min(len(task_keys), 64), replacement=True).tolist()
                for t_pos in drawn_tasks:
                    ms = tr_by_task[task_keys[t_pos]]
                    w_ms = weights[ms]
                    k_draw = min(len(ms), 8)
                    chosen = ([ms[j] for j in torch.multinomial(w_ms, k_draw, replacement=False).tolist()]
                              if float(w_ms.sum()) > 0 else ms[:k_draw])
                    order_list.extend(chosen)
                    if len(order_list) >= n:
                        break
            order = torch.tensor(order_list[:n], dtype=torch.long)
        else:
            order = tr_idx[torch.multinomial(sampler_weights, n, replacement=True)]
        total = 0.
        for start in range(0, n, cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            logits = head(x[idx].to(device))
            yb = y[idx].to(device)
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], yb)
            if cfg.pairwise_weight > 0.0:
                batch_tids = [tids[int(i)] for i in idx.tolist()]
                loss = loss + float(cfg.pairwise_weight) * pairwise_ranking_loss(logits[:, 0], yb, batch_tids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.clip_norm)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(idx)
        head.eval()
        with torch.no_grad():
            val_scores = head.success_probability(x[va_idx].to(device)).cpu()
        wt_auroc, wt_tasks = within_task_auroc(val_scores, y[va_idx], va_tids)
        record = {'epoch': epoch, 'train_loss': total / n, 'val_auroc': compute_auroc(val_scores, y[va_idx]),
                  'val_within_task_auroc': wt_auroc, 'val_within_task_n': wt_tasks,
                  'lr': scheduler.get_last_lr()[0]}
        log.write(json.dumps(record) + '\n')
        log.flush()
    log.close()
    with torch.no_grad():
        val_scores = head.success_probability(x[va_idx].to(device)).cpu()
        train_scores = head.success_probability(x[tr_idx].to(device)).cpu()
    generator = torch.Generator().manual_seed(seed + 1)
    permuted = y[va_idx][torch.randperm(len(va_idx), generator=generator)]
    wt_val, wt_n = within_task_auroc(val_scores, y[va_idx], va_tids)
    # episode-level AUROC: the mean score of an episode's boundaries
    episodes = {}
    for position, i in enumerate(va_idx.tolist()):
        episodes.setdefault(meta[i]['episode'], []).append((position, i))
    ep_scores = torch.tensor([float(val_scores[[p for p, _ in members]].mean()) for members in episodes.values()])
    ep_labels = torch.tensor([float(y[members[0][1]]) for members in episodes.values()])
    receipt = {'schema': T0_SCHEMA, 'stage': 'train', 'features': str(features_dir), 'config': cfg.__dict__,
               'rows': {'train': int(n), 'validation': int(len(va_idx)),
                        'train_success_share': float(y[tr_idx].mean()), 'validation_success_share': float(y[va_idx].mean()),
                        'validation_mixed_tasks': wt_n},
               'auroc': {'train': compute_auroc(train_scores, y[tr_idx]), 'validation': compute_auroc(val_scores, y[va_idx]),
                         'validation_within_task': wt_val,
                         'validation_label_permuted_control': compute_auroc(val_scores, permuted),
                         'validation_episode_mean': compute_auroc(ep_scores, ep_labels) if len(ep_labels) else float('nan'),
                         'validation_per_family': _per_family_auroc(val_scores, y[va_idx], [fam[i] for i in va_idx.tolist()])},
               'declared_success_criterion': {'validation_within_task_auroc_min': 0.70, 'validation_auroc_min': 0.72,
                                              'min_mixed_tasks': 10},
               'code': code_provenance(), 'seed': seed}
    # Gate branch@N on real within-task ranking quality (not cross-task difficulty confounding)
    receipt['criterion_met'] = (
        (wt_n >= 10 and not math.isnan(wt_val) and wt_val >= 0.70 and receipt['auroc']['validation'] >= 0.72)
        if wt_n > 0 else receipt['auroc']['validation'] >= 0.78
    )
    torch.save({'state_dict': head.state_dict(), 'hidden_dim': head.hidden_dim, 'mlp_dim': head.mlp_dim,
                'receipt': receipt}, out / 'head.pt')
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Recipe T0: step verifier on archived rollouts')
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--collection', action='append', required=True)
    b.add_argument('--policy-sha256', action='append', required=True,
                   help="One or more policy checkpoint SHA-256s, or 'any' to pool multiple nearby policies.")
    b.add_argument('--out', required=True)
    b.add_argument('--max-tokens', type=int, default=32768)
    b.add_argument('--max-rollouts', type=int)
    b.add_argument('--held-manifest')
    f = sub.add_parser('features')
    f.add_argument('--sequences', required=True)
    f.add_argument('--checkpoint', required=True)
    f.add_argument('--checkpoint-sha256', required=True)
    f.add_argument('--base', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--feature-mode', choices=('rich', 'last'), default='rich',
                   help='rich: [h_last, h_last - h_prompt_end, h_mean_completion, scalars] (default); last: legacy h_last.')
    t = sub.add_parser('train')
    t.add_argument('--features', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--epochs', type=int, default=VerifierTrainConfig.epochs)
    t.add_argument('--dropout', type=float, default=VerifierTrainConfig.dropout)
    t.add_argument('--weight-decay', type=float, default=VerifierTrainConfig.weight_decay)
    t.add_argument('--pairwise-weight', type=float, default=VerifierTrainConfig.pairwise_weight,
                   help='Within-task pairwise RankNet weight (default 0.25; 0 opts out to pure cross-task BCE).')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--progress-weighting', action=argparse.BooleanOptionalAction, default=True,
                   help='Upweight later episode boundaries (0.25 -> 1.0) where trajectories diverge (default: true).')
    args = parser.parse_args(argv)
    if args.command == 'build':
        policies = args.policy_sha256[0] if len(args.policy_sha256) == 1 else args.policy_sha256
        result = build_t0_sequences(args.collection, args.out, policy_sha256=policies,
                                    max_tokens=args.max_tokens, max_rollouts=args.max_rollouts,
                                    held_manifest=args.held_manifest)
        from pathlib import Path
        from ..common.code_provenance import code_provenance
        Path(args.out + '.json').write_text(json.dumps({'schema': T0_SCHEMA, 'stage': 'build', 'funnel': result,
                                                        'collections': args.collection,
                                                        'policy_sha256': policies,
                                                        'code': code_provenance()}, indent=2) + '\n')
    elif args.command == 'features':
        result = extract_t0_features(args.sequences, args.out, checkpoint=args.checkpoint, base=args.base,
                                     checkpoint_sha256=args.checkpoint_sha256, feature_mode=args.feature_mode)
    else:
        cfg = VerifierTrainConfig(epochs=args.epochs, dropout=args.dropout, weight_decay=args.weight_decay,
                                  pairwise_weight=args.pairwise_weight)
        result = train_t0(args.features, args.out, cfg=cfg, seed=args.seed,
                          device='cuda' if torch.cuda.is_available() else 'cpu',
                          progress_weighting=args.progress_weighting)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
