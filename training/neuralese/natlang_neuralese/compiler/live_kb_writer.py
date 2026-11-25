"""Live-KB experience writer for Recipe T2 (``plans/NATLANG_COMPILER.md`` §7.2 Stage C, Plan 06 Live-KB).

A small FP32 writer over the frozen policy's own token embeddings: experience texts -> ``(z_kb, k_kb)`` slots, call
contexts -> lookup queries ``q``, trained end to end through ``CompiledSiteHead`` by the native graph loss.

Scope, stated plainly (2026-11-25 integration review): this is a stand-in for hindsight-kb's ``ExperienceWriter``
(``hindsight_kb.models.experience``, the ``H`` the plan warm-starts from checkpoint 000000002). It is not that writer
and cannot load its weights: 000000002's ``writer.`` / ``query.`` tensors belong to a pplx-embed 0.6B encoder. The
first version advertised a warm start that read ``writer_warmstart.pt`` while the extractor writes ``writer_H.pt``,
and matched keys by name across different architectures, so it silently trained from scratch; it was removed.
Moving T2 onto hindsight-kb's ExperienceWriter (W0.2 extraction on g4a) is the open item. Slots written by this
writer live in its own key/value space: a ScopedExperienceKB is usable with it only when its records were written by
the same writer.
"""
from __future__ import annotations

import torch
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
):
    """Live-KB slots ``(z_kb [1, M, Dv], k_kb [1, M, Dk], w_kb [1, M], valid [1, M])`` for one training row.

    - Positives: same-family peers' verified experience, encoded live (gradients reach ``writer``). A peer that is the
      row itself, its task or a source-group sibling is never used: its experience text could carry this call's answer
      (``plans/NATLANG_COMPILER.md`` §9.4: related, non-identical peers, ``exclude_tasks`` enforced).
    - Distractors: other-family peers under the same exclusion, encoded without gradient.
    - ``null_restraint``: distractors only, at their real retrieval weights, so the head learns to leave memory that
      does not apply (plan: null-memory / already-solved examples train restraint). Zeroing the weights instead would
      only mask the slots and teach nothing.
    Weights are a differentiable function of query-key similarity, so the keys and queries train too."""
    device = writer.pool_query.device
    query_text = row.get('experience_query') or row.get('header') or row.get('binding') or ''
    q = writer.encode_query_texts(engine, [query_text])  # [1, Dk]
    family = row.get('family')
    usable = [p for p in peer_pool if p.get('experience_text') and not _shares_closure(p, row)]
    same_family = [p for p in usable if p.get('family') == family]
    diff_family = [p for p in usable if p.get('family') != family]
    pick = (lambda pool, k: rng.sample(pool, min(k, len(pool)))) if rng is not None else (lambda pool, k: pool[:k])
    pos_rows = [] if null_restraint else pick(same_family, k_pos)
    neg_rows = pick(diff_family, k_neg)
    z_parts, k_parts = [], []
    if pos_rows:
        z_live, k_live = writer.encode_experience_texts(engine, [p['experience_text'] for p in pos_rows])
        z_parts.append(z_live)
        k_parts.append(k_live)
    if neg_rows:
        with torch.no_grad():
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
