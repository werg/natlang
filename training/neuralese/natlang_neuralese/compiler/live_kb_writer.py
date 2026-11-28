"""Live-KB experience writer for Recipe T2 (``plans/NATLANG_COMPILER.md`` §7.2 Stage C, Plan 06 Live-KB).

A small FP32 writer over the frozen policy's own token embeddings: experience texts -> ``(z_kb, k_kb)`` slots, call
contexts -> lookup queries ``q``, trained end to end through ``CompiledSiteHead`` by the native graph loss.

Two writers, one interface (``encode_experience_texts`` -> ``(z_kb, k_kb)``, ``encode_query_texts`` -> ``q``,
``warm_parameters`` / ``fresh_parameters`` for the dual-LR groups):

- ``PplxWarmstartWriter`` (production, ``run_t2 train --writer-backend pplx``, the default): hindsight-kb's
  ``PPLXBlockEncoder`` writer ``H`` and query encoder ``Q`` (``pplx-embed-v2-late-0.6b``), warm-started from checkpoint
  000000002 (``hindsight-kb/scripts/extract_lfm3_warmstart.py`` -> ``writer_H.pt`` + ``query_Q.pt``, loaded strictly).
  The key space transfers; ``value_head`` is re-fit for LFM3 at the fresh LR (owner 2026-11-26: Stage C trains on it).
- ``LiveExperienceWriter``: a small FP32 pooler over the frozen policy's token embeddings, for CPU unit tests only
  (``run_t2`` refuses it on a GPU run). It cannot load 000000002's weights.

Slots written by a writer live in its own key/value space: a ScopedExperienceKB is usable with a writer only when its
records were written by the same writer.
"""
from __future__ import annotations

import torch
from pathlib import Path

from torch import nn
import torch.nn.functional as F


class LiveExperienceWriter(nn.Module):
    """Encodes experience texts into ``(z_kb [B, Dv], k_kb [B, Dk])`` and call contexts into ``q [B, Dk]`` (FP32)."""

    def __init__(
        self,
        embed_dim: int,
        *,
        value_dim: int = 256,
        key_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.value_dim = int(value_dim)
        self.key_dim = int(key_dim)
        self.hidden_dim = int(hidden_dim)

        self.in_norm = nn.LayerNorm(self.embed_dim)
        self.in_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(self.hidden_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.key_head = nn.Linear(self.hidden_dim, self.key_dim)
        self.query_head = nn.Linear(self.hidden_dim, self.key_dim)

        self.value_norm = nn.LayerNorm(self.hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.value_dim),
        )
        nn.init.normal_(self.value_head[-1].weight, std=0.02)
        nn.init.zeros_(self.value_head[-1].bias)
        self.float()

    def _encode_trunk(self, token_embeds: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool token embeddings ``[B, T, embed_dim]`` -> ``[B, hidden_dim]`` in FP32."""
        x = self.in_proj(self.in_norm(token_embeds.float()))
        q = self.pool_query.expand(x.shape[0], -1, -1)
        kv = self.attn_norm(x)
        pad_mask = None if mask is None else ~mask.to(torch.bool)
        pooled, _ = self.pool_attn(q, kv, kv, key_padding_mask=pad_mask, need_weights=False)
        h = (q + pooled).squeeze(1)
        h = h + self.ffn(self.ffn_norm(h))
        return h

    def encode_experience_embeds(self, token_embeds: torch.Tensor, mask: torch.Tensor | None = None):
        """Return ``(z_kb [B, Dv], k_kb [B, Dk] L2-normalized)``."""
        h = self._encode_trunk(token_embeds, mask=mask)
        k = F.normalize(self.key_head(h), dim=-1)
        z = self.value_head(self.value_norm(h))
        return z, k

    def encode_query_embeds(self, token_embeds: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return normalized lookup query ``q [B, Dk]``."""
        h = self._encode_trunk(token_embeds, mask=mask)
        return F.normalize(self.query_head(h), dim=-1)

    def embed_texts(self, engine, texts: list[str], *, max_tokens: int = 128):
        """Tokenize and look up frozen engine token embeddings (no grad through base embeddings)."""
        device = self.pool_query.device
        emb_weight = _engine_embedding_weight(engine)
        tok_lists = [_tokenize_text(engine, t, max_tokens=max_tokens) for t in texts]
        max_len = max(1, max(len(ids) for ids in tok_lists))
        ids = torch.zeros((len(texts), max_len), dtype=torch.long, device=emb_weight.device)
        mask = torch.zeros((len(texts), max_len), dtype=torch.bool, device=device)
        vocab = emb_weight.shape[0]
        for row_idx, seq in enumerate(tok_lists):
            clamped = [min(max(0, int(t)), vocab - 1) for t in seq]
            ids[row_idx, :len(clamped)] = torch.tensor(clamped, dtype=torch.long, device=emb_weight.device)
            mask[row_idx, :len(clamped)] = True
        with torch.no_grad():
            embeds = F.embedding(ids, emb_weight).detach().to(device=device, dtype=torch.float32)
        return embeds, mask

    def encode_experience_texts(self, engine, texts: list[str], *, max_tokens: int = 128):
        embeds, mask = self.embed_texts(engine, texts, max_tokens=max_tokens)
        return self.encode_experience_embeds(embeds, mask=mask)

    def encode_query_texts(self, engine, texts: list[str], *, max_tokens: int = 128) -> torch.Tensor:
        embeds, mask = self.embed_texts(engine, texts, max_tokens=max_tokens)
        return self.encode_query_embeds(embeds, mask=mask)

    def encode_experience_blocks(self, engine, texts: list[str], *, max_tokens: int = 128, max_slots: int = 8):
        """Multi-block interface parity with ``PplxWarmstartWriter`` (returns list of ``(keys [M, Dk], vals [M, Dv])``)."""
        z, k = self.encode_experience_texts(engine, texts, max_tokens=max_tokens)
        return [(k[i:i + 1], z[i:i + 1]) for i in range(len(texts))]

    def encode_query_blocks(self, engine, texts: list[str], *, max_tokens: int = 128):
        q = self.encode_query_texts(engine, texts, max_tokens=max_tokens)
        return [q[i:i + 1] for i in range(len(texts))]

    def warm_parameters(self) -> list[nn.Parameter]:
        return []

    def fresh_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())


class PplxWarmstartWriter(nn.Module):
    """Production Live-KB writer wrapping hindsight-kb's ``PPLXBlockEncoder`` (`pplx-embed-v2-late-0.6b`)
    warm-started from checkpoint ``000000002`` (`writer_H.pt` + `query_Q.pt`).

    Exposes exact dual-LR parameter groups:
    - ``warm_parameters()``: warm-started ``H_key``/backbone & ``Q`` parameters (trained at ``warm_lr = 2e-5``).
    - ``fresh_parameters()``: ``writer.value_head`` (re-fit for LFM3-1.7B at ``lr = 2e-4``) + FP32 slot pooler.
    """

    def __init__(self, writer_encoder: nn.Module, query_encoder: nn.Module, *, value_dim: int = 256, key_dim: int = 128):
        super().__init__()
        self.writer_encoder = writer_encoder
        self.query_encoder = query_encoder
        self.value_dim = int(value_dim)
        self.key_dim = int(key_dim)
        # Lightweight FP32 learned attention pooler (zero-init residual over L2-norm salience so step 0 preserves
        # 000000002's sharp trained block keys rather than washing them out with a random query vector).
        self.pool_query = nn.Parameter(torch.zeros(1, self.key_dim, dtype=torch.float32))
        self.query_pool = nn.Parameter(torch.zeros(1, self.key_dim, dtype=torch.float32))

    @classmethod
    def from_warmstart(
        cls,
        *,
        revision: str,
        warmstart_dir: str | None = None,
        model_id: str = 'perplexity-ai/pplx-embed-v2-late-0.6b',
        value_dim: int = 256,
        key_dim: int = 128,
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        use_hf_kernels: bool | None = None,
    ) -> 'PplxWarmstartWriter':
        """``use_hf_kernels`` (default: on for CUDA): hindsight-kb's differentiable FLA / causal-conv / Liger kernels for
        the Qwen3.5 backbone, the configuration 000000002 was trained with (configs/joint-colab.json); without them
        transformers runs its reference PyTorch linear attention. ``False`` opts out."""
        if warmstart_dir is not None:
            root = Path(warmstart_dir)
            writer_path = root / 'writer_H.pt'
            query_path = root / 'query_Q.pt'
            if not writer_path.is_file() or not query_path.is_file():
                raise FileNotFoundError(
                    f'PplxWarmstartWriter requires writer_H.pt and query_Q.pt in {root} '
                    '(run hindsight-kb/scripts/extract_lfm3_warmstart.py first)'
                )

        from . import hkb_dependency
        hkb_dependency.require()
        from hindsight_kb.integrations.pplx import PPLXBlockEncoder

        if use_hf_kernels is None:
            use_hf_kernels = str(device).startswith('cuda')
        writer_enc = PPLXBlockEncoder.from_pretrained(
            revision=revision, model_id=model_id, writer=True, value_dim=value_dim, device=device, dtype=dtype,
            use_hf_kernels=use_hf_kernels,
        )
        query_enc = PPLXBlockEncoder.from_pretrained(
            revision=revision, model_id=model_id, writer=False, value_dim=value_dim, device=device, dtype=dtype,
            use_hf_kernels=use_hf_kernels,
        )
        writer_enc.checkpoint_views = True
        query_enc.checkpoint_views = True
        # PplxWarmstartWriter.train() keeps both encoders in eval() mode (no dropout), so PPLXBlockEncoder only
        # checkpoints view forwards during grad-enabled passes when checkpoint_eval_gradients is explicitly True.
        writer_enc.checkpoint_eval_gradients = True
        query_enc.checkpoint_eval_gradients = True

        if warmstart_dir is not None:
            w_sd = torch.load(writer_path, map_location='cpu', weights_only=True)
            q_sd = torch.load(query_path, map_location='cpu', weights_only=True)
            # Strict load: never silently train from scratch on mismatched keys/shapes
            writer_enc.load_state_dict(w_sd, strict=True)
            query_enc.load_state_dict(q_sd, strict=True)

        # The encoders compute in their serving dtype (BF16) and keep it as their parameter dtype; the trainer's
        # optimizer holds FP32 master weights and moments for them (hindsight-kb MasterAdamW, run_t2). A value_head cast
        # to FP32 alone fed BF16 hidden states into an FP32 Linear (a dtype error on the first encode).
        projection = getattr(writer_enc, 'projection', None)
        if projection is not None and int(projection.out_features) != int(key_dim):
            raise ValueError(f'pplx key projection is {projection.out_features}-d; the compiled-site config needs key_dim={key_dim}')
        inst = cls(writer_enc, query_enc, value_dim=value_dim, key_dim=key_dim)
        inst.warmstart = None if warmstart_dir is None else _warmstart_manifest(Path(warmstart_dir), writer_path, query_path)
        inst.pool_query.data = inst.pool_query.data.to(device)
        inst.query_pool.data = inst.query_pool.data.to(device)
        return inst

    def train(self, mode: bool = True):
        """The encoders stay in eval mode (deterministic encoding, no dropout) while their parameters train, as
        hindsight-kb's experience trainer runs its writer (``writer.eval().requires_grad_(True)``)."""
        super().train(mode)
        self.writer_encoder.eval()
        self.query_encoder.eval()
        return self

    def _pool_kv(self, keys: torch.Tensor, values: torch.Tensor | None, pool_vec: torch.Tensor):
        kf = keys.float()
        # Preserve native block salience at step 0 (when pool_vec == 0) via log-norm weighting + learned query bias.
        salience = torch.linalg.vector_norm(kf, dim=-1).clamp_min(1e-6).log()
        logits = salience + (F.normalize(kf, dim=-1) @ pool_vec.float().T).squeeze(-1)
        weights = logits.softmax(dim=0)
        k_out = F.normalize((weights[:, None] * kf).sum(dim=0), dim=-1)
        if values is None:
            return k_out, None
        v_out = (weights[:, None] * values.float()).sum(dim=0)
        return k_out, v_out

    def encode_experience_blocks(
        self, engine, texts: list[str], *, max_tokens: int = 512, max_slots: int = 8,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Preserve 000000002's native multi-block keys and values (up to ``max_slots`` highest-salience blocks/text)
        without collapsing them into a single pooled slot."""
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for text in texts:
            blocks = self.writer_encoder.encode_blocks([text], 'full')
            ks = torch.cat([b.keys.float() for b in blocks if len(b.keys)], dim=0)
            vs = torch.cat([b.values.float() for b in blocks if len(b.keys)], dim=0)
            if max_slots > 0 and ks.shape[0] > max_slots:
                top_idx = torch.linalg.vector_norm(ks, dim=-1).topk(int(max_slots)).indices.sort().values
                ks = ks[top_idx]
                vs = vs[top_idx]
            out.append((F.normalize(ks, dim=-1), vs))
        return out

    def encode_query_blocks(self, engine, texts: list[str], *, max_tokens: int = 512) -> list[torch.Tensor]:
        """Return native multi-block normalized query keys ``[M, Dk]`` per text from 000000002's ``query_Q``."""
        out: list[torch.Tensor] = []
        for text in texts:
            blocks = self.query_encoder.encode_blocks([text], 'forward')
            ks = torch.cat([b.keys.float() for b in blocks if len(b.keys)], dim=0)
            out.append(F.normalize(ks, dim=-1))
        return out

    def encode_experience_texts(self, engine, texts: list[str], *, max_tokens: int = 512):
        """Encode each experience text via ``PPLXBlockEncoder`` -> ``(z_kb [B, Dv], k_kb [B, Dk])`` in FP32."""
        z_list, k_list = [], []
        for text in texts:
            blocks = self.writer_encoder.encode_blocks([text], 'full')
            ks = torch.cat([b.keys for b in blocks if len(b.keys)], dim=0)
            vs = torch.cat([b.values for b in blocks if len(b.keys)], dim=0)
            k_one, v_one = self._pool_kv(ks, vs, self.pool_query)
            k_list.append(k_one)
            z_list.append(v_one)
        return torch.stack(z_list, dim=0), torch.stack(k_list, dim=0)

    def encode_query_texts(self, engine, texts: list[str], *, max_tokens: int = 512) -> torch.Tensor:
        q_list = []
        for text in texts:
            blocks = self.query_encoder.encode_blocks([text], 'forward')
            ks = torch.cat([b.keys for b in blocks if len(b.keys)], dim=0)
            q_one, _ = self._pool_kv(ks, None, self.query_pool)
            q_list.append(q_one)
        return torch.stack(q_list, dim=0)

    def fresh_parameters(self) -> list[nn.Parameter]:
        """Re-fit ``value_head`` and pool vectors at fresh LR (2e-4) for LFM3-1.7B."""
        fresh = [self.pool_query, self.query_pool]
        if getattr(self.writer_encoder, 'value_head', None) is not None:
            fresh.extend(self.writer_encoder.value_head.parameters())
        return fresh

    def encode_experience_blocks(self, engine, texts: list[str], *, max_tokens: int = 128, max_slots: int = 8):
        """Multi-block interface parity with ``PplxWarmstartWriter`` (returns list of ``(keys [M, Dk], vals [M, Dv])``)."""
        z, k = self.encode_experience_texts(engine, texts, max_tokens=max_tokens)
        return [(k[i:i + 1], z[i:i + 1]) for i in range(len(texts))]

    def encode_query_blocks(self, engine, texts: list[str], *, max_tokens: int = 128):
        q = self.encode_query_texts(engine, texts, max_tokens=max_tokens)
        return [q[i:i + 1] for i in range(len(texts))]

    def warm_parameters(self) -> list[nn.Parameter]:
        """Warm-started 000000002 ``H_key``/backbone & ``Q`` parameters trained at ``warm_lr`` (2e-5)."""
        fresh_ids = {id(p) for p in self.fresh_parameters()}
        return [p for p in self.parameters() if id(p) not in fresh_ids]

def _warmstart_manifest(root, writer_path, query_path) -> dict:
    """What the warm start loaded: the extractor's manifest (when present) and the loaded files' SHA-256."""
    import json
    from ..common.hashing import sha256_file_hex as sha

    manifest_path = root / 'warmstart-manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    loaded = {'writer_sha256': sha(writer_path), 'query_sha256': sha(query_path)}
    if manifest is not None and (manifest.get('writer_sha256'), manifest.get('query_sha256')) != (loaded['writer_sha256'], loaded['query_sha256']):
        raise ValueError(f'{root}: writer_H.pt / query_Q.pt do not match warmstart-manifest.json')
    return {'dir': str(root), **loaded, 'manifest': manifest}


def _engine_embedding_weight(engine) -> torch.Tensor:
    port = engine.shared_model.port
    if hasattr(port, 'input_embedding_weight'):
        w = port.input_embedding_weight
        if isinstance(w, torch.Tensor):
            return w
    for name, param in engine.shared_model.named_parameters():
        if 'embed_tokens' in name or name.endswith('wte.weight'):
            return param
    if hasattr(port, 'output_weight') and isinstance(port.output_weight, torch.Tensor):
        return port.output_weight
    raise ValueError('could not locate token embedding weight on engine.shared_model')


def _tokenize_text(engine, text: str, *, max_tokens: int = 128) -> list[int]:
    """The policy tokenizer's ids (no fallback: bytes as token ids would be a different, meaningless input)."""
    ids = engine.tokenizer(text, add_special_tokens=False)['input_ids'][:max_tokens]
    if not ids:
        raise ValueError('an experience text tokenized to no tokens')
    return [int(i) for i in ids]


def _shares_closure(peer, row):
    """Whether ``peer`` is ``row``'s own task or a source-group sibling (its experience could carry the answer)."""
    if peer.get('task_id') == row.get('task_id') or peer.get('key') == row.get('key'):
        return True
    return bool(set(peer.get('source_groups') or ()) & set(row.get('source_groups') or ()))


def related_peer(peer, row):
    """Whether ``peer`` is a labelled relative of ``row``: the same task family or the same compiled site."""
    return (peer.get('family') == row.get('family')
            or (row.get('site') is not None and peer.get('site') == row.get('site')))


class KbIndex:
    """The writer's keys of a peer pool, computed without gradient and refreshed by the trainer (a stale retrieval
    index, as retrieval-augmented trainers keep one): ``build_live_kb_slots`` fills positives by query similarity from
    it where a row has too few labelled relatives."""

    def __init__(self, engine, writer, pool, *, batch=32):
        self.keys = {}
        texts = [(p['key'], p['experience_text']) for p in pool if p.get('experience_text')]
        was_training = writer.training
        writer.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch):
                part = texts[start:start + batch]
                _, k = writer.encode_experience_texts(engine, [text for _, text in part])
                for (key, _), value in zip(part, k.float()):
                    self.keys[key] = value.detach()
        writer.train(was_training)

    def nearest(self, query, peers, k):
        """The ``k`` peers whose indexed key is most similar to ``query`` ([1, Dk] or [Dk], detached)."""
        peers = [p for p in peers if p['key'] in self.keys]
        if not peers or k <= 0:
            return []
        q = query.detach().float().reshape(-1)
        keys = torch.stack([self.keys[p['key']].to(q.device) for p in peers])
        order = torch.argsort(keys @ q, descending=True)[:k].tolist()
        return [peers[i] for i in order]


def build_live_kb_slots(
    engine,
    writer: LiveExperienceWriter,
    row: dict,
    peer_pool: list[dict],
    *,
    k_pos: int = 2,
    k_neg: int = 2,
    temperature: float = 0.25,
    sim_threshold: float = -0.25,
    null_restraint: bool = False,
    rng=None,
    index: KbIndex | None = None,
    neg_grad: bool = True,
    chosen: list | None = None,
):
    """Live-KB slots ``(z_kb [1, M, Dv], k_kb [1, M, Dk], w_kb [1, M], valid [1, M])`` for one training row.

    - Positives: related peers' verified experience (``related_peer``: same task family or same site), encoded live
      (gradients reach ``writer``). With an ``index``, a row with fewer than ``k_pos`` relatives is filled with the
      peers nearest its query in the index (2026-11-26: 71% of T2 train rows had no same-family peer and saw no
      positive at all; retrieval by similarity is how the KB is read at serving). A peer that is the row itself, its
      task or a source-group sibling is never used: its experience text could carry this call's answer
      (``plans/NATLANG_COMPILER.md`` §9.4: related, non-identical peers, ``exclude_tasks`` enforced).
    - Distractors: unrelated peers under the same exclusion, encoded with gradient (``neg_grad`` False encodes them
      without), so the writer's keys also learn to score down what does not help.
    - ``null_restraint``: distractors only, at their real retrieval weights, so the head learns to leave memory that
      does not apply (plan: null-memory / already-solved examples train restraint). Zeroing the weights instead would
      only mask the slots and teach nothing.
    - ``rng`` draws relatives and distractors (None: the first in pool order). ``chosen``, when given, receives the
      keys of the peers used, positives first.
    Weights are a differentiable function of query-key similarity, so the keys and queries train too."""
    device = writer.pool_query.device
    query_text = row.get('experience_query') or row.get('header') or row.get('binding') or ''
    q = writer.encode_query_texts(engine, [query_text])  # [1, Dk]
    usable = [p for p in peer_pool if p.get('experience_text') and not _shares_closure(p, row)]
    related = [p for p in usable if related_peer(p, row)]
    unrelated = [p for p in usable if not related_peer(p, row)]
    pick = (lambda pool, k: rng.sample(pool, min(k, len(pool)))) if rng is not None else (lambda pool, k: pool[:k])
    pos_rows = [] if null_restraint else pick(related, k_pos)
    if not null_restraint and index is not None and len(pos_rows) < k_pos:
        pos_rows += index.nearest(q, unrelated, k_pos - len(pos_rows))
    taken = {p['key'] for p in pos_rows}
    neg_rows = pick([p for p in unrelated if p['key'] not in taken], k_neg)
    if chosen is not None:
        chosen.extend(p['key'] for p in [*pos_rows, *neg_rows])
    z_parts, k_parts = [], []
    if pos_rows:
        z_live, k_live = writer.encode_experience_texts(engine, [p['experience_text'] for p in pos_rows])
        z_parts.append(z_live)
        k_parts.append(k_live)
    if neg_rows:
        with torch.set_grad_enabled(torch.is_grad_enabled() and neg_grad):
            z_neg, k_neg_t = writer.encode_experience_texts(engine, [p['experience_text'] for p in neg_rows])
        z_parts.append(z_neg)
        k_parts.append(k_neg_t)
    if not z_parts:
        return (torch.zeros((1, 0, writer.value_dim), device=device), torch.zeros((1, 0, writer.key_dim), device=device),
                torch.zeros((1, 0), device=device), torch.zeros((1, 0), device=device, dtype=torch.bool))
    z_all = torch.cat(z_parts, dim=0).unsqueeze(0)  # [1, M, Dv]
    k_all = torch.cat(k_parts, dim=0).unsqueeze(0)  # [1, M, Dk]
    sims = (q.unsqueeze(1) * k_all).sum(dim=-1)     # [1, M]
    w_all = F.softplus((sims - float(sim_threshold)) / max(1e-3, float(temperature)))
    valid = torch.ones((1, z_all.shape[1]), device=device, dtype=torch.bool)
    return z_all, k_all, w_all, valid
