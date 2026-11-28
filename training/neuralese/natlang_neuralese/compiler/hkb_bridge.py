"""The compiler's scoped experience KB (`plans/NATLANG_COMPILER.md` §7, W4.1): scopes and held admission over
hindsight-kb's store and index.

Storage, the index (exact, HNSW or IVF), task provenance and retrieval are hindsight-kb's (``kb.sqlite.SQLiteKB``,
``kb.provenance``, ``retrieval.Retriever.search_scopes``); this module adds only what the compiler needs on top:

- **Scopes.** ``natlang:global``, ``group:<domain>``, ``repo:<sha>`` and ``fn:<site_id>`` are hindsight-kb record
  scopes (its namespaces). A retrieval over several scopes runs each scope's indexed top-k and mixes them in one
  rescoring.
- **Held admission.** ``HeldAdmission`` resolves the closure training must never take
  (``rl.held_eval.training_exclusion_ids``: held-v3/v4, their reservations and siblings by split and source group) over
  the task rows the records come from. A record of an excluded task, or of a task outside those rows (whose closure
  cannot be checked), is refused; opening a KB audits every stored task the same way, so a held set registered after
  admission closes the KB until its records are removed.
- **Retrieval-time exclusion.** ``exclude_tasks`` (the current task and its siblings) goes to hindsight-kb's provenance
  index and is applied before each scope's top-k, to records and relations alike.

Stage A reads ``retrieve`` (records with their demonstration text, weights and the evidence gate); Stage C reads
``retrieve_unified_slots`` (``z_kb [1, M, Dv]``, ``k_kb [1, M, Dk]``, ``w [1, M]``, ``valid [1, M]``).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from . import hkb_dependency

hkb_dependency.require()
from hindsight_kb.config import RetrievalConfig  # noqa: E402
from hindsight_kb.kb.base import Record  # noqa: E402
from hindsight_kb.kb.provenance import KIND, STRENGTH, TASKS  # noqa: E402
from hindsight_kb.kb.sqlite import SQLiteKB  # noqa: E402
from hindsight_kb.retrieval import Retriever  # noqa: E402

SCOPE = re.compile(r'natlang:global|(group|repo|fn):[^\s\x00]+')
KINDS = ('success', 'repair', 'doc', 'guidance')
META = 'natlang_experience'   # the compiler's own fields inside a hindsight-kb record's metadata


def check_scope(scope: str) -> str:
    if not isinstance(scope, str) or not SCOPE.fullmatch(scope):
        raise ValueError(f'an experience scope is natlang:global, group:<domain>, repo:<sha> or fn:<site_id>, '
                         f'not {scope!r}')
    return scope


def record_key(scope: str, record_id: str) -> int:
    """hindsight-kb's integer record id for a compiler record: a 62-bit hash of (scope, record id)."""
    digest = hashlib.sha256(f'{scope}\x00{record_id}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') >> 2


@dataclass(frozen=True)
class ExperienceEntry:
    record_id: str
    scope: str
    key: torch.Tensor                  # [D_k]
    value: torch.Tensor                # [D_v] (256-d semantic slot)
    text: str                          # Stage A demonstration / repair lesson
    task_ids: tuple[str, ...]
    strength: float = 1.0              # (0, 1]; 0 is not admitted
    kind: str = 'success'              # 'success' | 'repair' | 'doc' | 'guidance'
    compacted_state_id: str | None = None


@dataclass(frozen=True)
class RetrievedExperience:
    record_id: str
    scope: str
    text: str
    kind: str
    task_ids: tuple[str, ...]
    compacted_state_id: str | None
    similarity: float
    weight: float                      # raw evidence / (null mass + total kept evidence)
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class HeldAdmission:
    """The task rows records may come from, and the ids among them training must never take."""
    known: frozenset[str]
    excluded: frozenset[str]

    @classmethod
    def from_rows(cls, rows, held_manifest=None, *, extra=(), repo=None) -> 'HeldAdmission':
        """``rows``: the pool rows (or a JSONL path) the records' tasks are drawn from; the exclusion is
        ``held_eval.training_exclusion_ids(rows, held_manifest, extra=extra, repo=repo)``."""
        from ..rl import held_eval
        if isinstance(rows, (str, Path)):
            rows = list(held_eval.read_jsonl(rows))
        rows = list(rows)
        excluded = held_eval.training_exclusion_ids(rows, held_manifest, extra=extra, repo=repo)
        return cls(frozenset(row['id'] for row in rows), frozenset(excluded))

    def check(self, task_ids, *, what='record') -> None:
        for task in task_ids:
            if task in self.excluded:
                raise ValueError(f'refusing {what}: task {task!r} is held or shares a held split/source group')
            if task not in self.known:
                raise ValueError(f'refusing {what}: task {task!r} is not among the admission rows, so its held '
                                 'closure cannot be checked; build HeldAdmission.from_rows over the rows it comes from')


class ScopedExperienceKB:
    """The four experience scopes over one hindsight-kb SQLite KB (``path``), with held admission."""

    def __init__(self, path, *, admission: HeldAdmission, key_dim: int = 128, value_dim: int = 256,
                 index: str = 'exact', readonly: bool = False, **kb_options):
        if not isinstance(admission, HeldAdmission):
            raise TypeError('ScopedExperienceKB needs a HeldAdmission (HeldAdmission.from_rows over the task rows)')
        self.admission = admission
        self.key_dim, self.value_dim = int(key_dim), int(value_dim)
        self.kb = SQLiteKB(path, self.key_dim, self.value_dim, index=index, readonly=readonly, **kb_options)
        self._readonly = bool(readonly)
        try:
            self._audit()
            if not self._readonly:
                self._ensure_outcomes_table()
        except BaseException:
            self.kb.close()
            raise

    def _ensure_outcomes_table(self) -> None:
        self.kb.conn.execute(
            'CREATE TABLE IF NOT EXISTS experience_outcomes ('
            'scope TEXT NOT NULL, record_id TEXT NOT NULL, injections INTEGER NOT NULL, '
            'lift_sum REAL NOT NULL, ema_lift REAL NOT NULL, last_reward REAL NOT NULL, '
            'PRIMARY KEY (scope, record_id))'
        )
        self.kb.conn.execute(
            'CREATE TABLE IF NOT EXISTS experience_drift ('
            'scope TEXT NOT NULL, record_id TEXT NOT NULL, drift_step INTEGER NOT NULL, '
            'reproduced_step INTEGER NOT NULL, drift_norm REAL NOT NULL, '
            'PRIMARY KEY (scope, record_id))'
        )
        self.kb.conn.commit()

    def _audit(self) -> None:
        if not self.kb.conn.execute("SELECT 1 FROM sqlite_master WHERE name='record_tasks'").fetchone():
            if len(self.kb):
                raise ValueError(f'{self.kb.path} has no task provenance index; its records cannot be held-checked')
            return
        stored = [row[0] for row in self.kb.conn.execute('SELECT DISTINCT task_id FROM record_tasks ORDER BY task_id')]
        self.admission.check(stored, what=f'to open {self.kb.path}')

    def __len__(self) -> int:
        return len(self.kb)

    def close(self) -> None:
        self.kb.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def provenance(self) -> dict:
        return {'hindsight_kb': hkb_dependency.provenance(), 'path': str(self.kb.path),
                'database_id': self.kb.database_id, 'generation': self.kb.generation}

    # ---- admission ---------------------------------------------------------------------------------------------------
    def _record(self, entry: ExperienceEntry) -> Record:
        check_scope(entry.scope)
        if entry.kind not in KINDS:
            raise ValueError(f'experience kind is one of {KINDS}, not {entry.kind!r}')
        if not 0.0 < float(entry.strength) <= 1.0:
            raise ValueError(f'experience strength is in (0, 1], not {entry.strength!r}')
        if entry.key.shape != (self.key_dim,) or entry.value.shape != (self.value_dim,):
            raise ValueError(f'expected key ({self.key_dim},) and value ({self.value_dim},), '
                             f'got {tuple(entry.key.shape)} and {tuple(entry.value.shape)}')
        tasks = tuple(entry.task_ids)
        self.admission.check(tasks, what=f'record {entry.record_id!r}')
        metadata = {KIND: 'factual' if entry.kind == 'doc' else 'experience', STRENGTH: float(entry.strength),
                    TASKS: sorted(set(tasks)),
                    META: {'record_id': entry.record_id, 'kind': entry.kind, 'text': entry.text,
                           'compacted_state_id': entry.compacted_state_id}}
        return Record(record_key(entry.scope, entry.record_id), entry.key.detach().float().cpu(),
                      entry.value.detach().float().cpu(), scope=entry.scope, metadata=metadata)

    def add_many(self, entries) -> int:
        """Admit ``entries`` in one transaction (all or none); a strength of 0 is skipped. Returns the number written."""
        records = [self._record(e) for e in entries if float(e.strength) > 0.0]
        for scope in {r.scope for r in records}:
            mine = [r for r in records if r.scope == scope]
            for old in self.kb.fetch_records([r.id for r in mine], scope=scope):
                new = next(r for r in mine if r.id == old.id)
                if old.metadata.get(META, {}).get('record_id') != new.metadata[META]['record_id']:
                    raise ValueError(f'record id collision in {scope}: {new.metadata[META]["record_id"]!r}')
        if records:
            self.kb.upsert(records)
        return len(records)

    def add(self, entry: ExperienceEntry) -> bool:
        return self.add_many([entry]) == 1

    # ---- retrieval ---------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _search(self, query_key, *, scopes, exclude_tasks, top_k, threshold, null_mass):
        scopes = tuple(check_scope(s) for s in dict.fromkeys(scopes))
        top_k = int(top_k)
        if query_key.shape != (self.key_dim,):
            raise ValueError(f'expected a query key ({self.key_dim},), got {tuple(query_key.shape)}')
        if not scopes or top_k < 1:
            return None, scopes
        retriever = Retriever(RetrievalConfig(top_k=top_k, threshold=float(threshold), null_mass=float(null_mass),
                                              slots=1))
        evidence = retriever.search_scopes(query_key.detach().float().cpu()[None], self.kb, scopes=scopes,
                                           exclude_tasks=tuple(sorted(set(exclude_tasks))), memory_records=top_k)
        return evidence.records, scopes

    def retrieve(self, query_key: torch.Tensor, *, scopes, exclude_tasks=(), top_k: int = 16, threshold: float = 0.2,
                 null_mass: float = 0.1) -> tuple[list[RetrievedExperience], float]:
        """Stage A: the top-k records of ``scopes`` (tasks in ``exclude_tasks`` removed before selection) by
        hindsight-kb's thresholded evidence, with weights normalized against the null mass, and the evidence gate."""
        records, scopes = self._search(query_key, scopes=scopes, exclude_tasks=exclude_tasks, top_k=top_k,
                                       threshold=threshold, null_mass=null_mass)
        if records is None or not bool(records.valid.any()):
            return [], 0.0
        n = int(records.valid.sum())
        ids = records.ids[:n].tolist()
        raw = records.weights[:n].float()
        total = float(raw.sum())
        denom = float(null_mass) + total
        found = {}
        for scope in scopes:
            for rec in self.kb.fetch_records(ids, scope=scope):
                found[rec.id] = rec
        q = F.normalize(query_key.detach().float().cpu(), dim=-1)
        hits = []
        for i, rid in enumerate(ids):
            rec = found[rid]
            meta = rec.metadata[META]
            hits.append(RetrievedExperience(
                record_id=meta['record_id'], scope=rec.scope, text=meta['text'], kind=meta['kind'],
                task_ids=tuple(rec.metadata.get(TASKS, ())), compacted_state_id=meta.get('compacted_state_id'),
                similarity=float(q @ F.normalize(rec.key.float(), dim=-1)), weight=float(raw[i]) / denom,
                key=records.keys[i].detach().float(), value=records.values[i].detach().float()))
        return hits, total / denom

    def retrieve_unified_slots(self, query_key: torch.Tensor, *, scopes, exclude_tasks=(), max_slots: int = 16,
                               threshold: float = 0.2, null_mass: float = 0.1, device: torch.device | str = 'cpu',
                               normalize: bool = True):
        """Stage C: padded ``(z_kb [1, M, Dv], k_kb [1, M, Dk], weights [1, M], valid [1, M])``, M = ``max_slots``.

        Pass ``normalize=False`` when feeding directly into ``CompiledSiteHead._assemble_unified_slots``, which
        applies ``null_mass`` itself (avoiding double null-mass gating)."""
        records, _ = self._search(query_key, scopes=scopes, exclude_tasks=exclude_tasks, top_k=max_slots,
                                  threshold=threshold, null_mass=null_mass)
        m = int(max_slots)
        z = torch.zeros((1, m, self.value_dim), dtype=torch.float32, device=device)
        k = torch.zeros((1, m, self.key_dim), dtype=torch.float32, device=device)
        w = torch.zeros((1, m), dtype=torch.float32, device=device)
        valid = torch.zeros((1, m), dtype=torch.bool, device=device)
        if records is None:
            return z, k, w, valid
        raw = records.weights.float() * records.valid
        denom = float(null_mass) + float(raw.sum())
        z[0] = records.values.float().to(device)
        k[0] = records.keys.float().to(device)
        w[0] = ((raw / denom) if normalize else raw).to(device)
        valid[0] = records.valid.to(device)
        return z, k, w, valid

    # ---- Stage-B Neuralese payloads & outcome-driven credit assignment ----------------------------------------------
    def retrieve_neuralese_payloads(
        self,
        query_key: torch.Tensor,
        *,
        scopes,
        exclude_tasks=(),
        top_k: int = 8,
        threshold: float = 0.2,
        null_mass: float = 0.1,
    ) -> dict:
        """Stage A/B retrieval returning Neuralese ``compacted_state_id`` blocks (served today on all 4 engines)
        plus concise text fallbacks and ``used`` ``(scope, record_id)`` pairs for ``record_outcomes``."""
        hits, gate = self.retrieve(
            query_key, scopes=scopes, exclude_tasks=exclude_tasks,
            top_k=top_k, threshold=threshold, null_mass=null_mass,
        )
        blocks = []
        text_fallbacks = []
        used = []
        for hit in hits:
            used.append((hit.scope, hit.record_id))
            item = {
                'record_id': hit.record_id,
                'scope': hit.scope,
                'kind': hit.kind,
                'weight': round(float(hit.weight), 6),
                'similarity': round(float(hit.similarity), 6),
            }
            if hit.compacted_state_id:
                blocks.append({**item, 'compacted_state_id': hit.compacted_state_id})
            else:
                text_fallbacks.append({**item, 'text': hit.text})
        return {'gate': round(float(gate), 6), 'blocks': blocks, 'text_fallbacks': text_fallbacks, 'used': used}

    def record_outcomes(
        self,
        used: list[tuple[str, str]],
        *,
        reward: float,
        baseline_reward: float,
        lr: float = 0.25,
        ema_alpha: float = 0.2,
        min_strength: float = 0.05,
    ) -> dict[str, dict[str, float]]:
        """Counterfactual outcome credit assignment: update EMA lift and live record ``strength`` in SQLite.

        Positive counterfactual lift (``reward > baseline_reward``) increases the record's retrieval strength (up to
        1.0); zero or negative lift decays it toward ``min_strength`` so weak records naturally drop out of top-k
        selection and become candidates for ``prune_weak``.
        """
        if self._readonly:
            raise ValueError('cannot record outcomes on a readonly ScopedExperienceKB')
        if not used:
            return {}
        lift = float(reward) - float(baseline_reward)
        updates: list[Record] = []
        summary: dict[str, dict[str, float]] = {}
        seen: set[tuple[str, str]] = set()
        for scope, rid in used:
            key_pair = (check_scope(scope), str(rid))
            if key_pair in seen:
                continue
            seen.add(key_pair)
            int_id = record_key(*key_pair)
            fetched = self.kb.fetch_records([int_id], scope=key_pair[0])
            if not fetched:
                continue
            rec = fetched[0]
            meta = dict(rec.metadata)
            old_strength = float(meta.get(STRENGTH, 1.0))
            new_strength = min(1.0, max(float(min_strength), old_strength * math.exp(float(lr) * lift)))
            meta[STRENGTH] = new_strength
            updates.append(Record(rec.id, rec.key, rec.value, scope=rec.scope, metadata=meta))
            row = self.kb.conn.execute(
                'SELECT injections, lift_sum, ema_lift FROM experience_outcomes WHERE scope=? AND record_id=?',
                key_pair,
            ).fetchone()
            if row is None:
                inj, lift_sum, ema_lift = 1, lift, lift
            else:
                inj = int(row[0]) + 1
                lift_sum = float(row[1]) + lift
                ema_lift = (1.0 - float(ema_alpha)) * float(row[2]) + float(ema_alpha) * lift
            self.kb.conn.execute(
                'INSERT OR REPLACE INTO experience_outcomes '
                '(scope, record_id, injections, lift_sum, ema_lift, last_reward) VALUES (?, ?, ?, ?, ?, ?)',
                (key_pair[0], key_pair[1], inj, lift_sum, ema_lift, float(reward)),
            )
            summary[f'{key_pair[0]}:{key_pair[1]}'] = {
                'injections': float(inj),
                'ema_lift': round(ema_lift, 6),
                'old_strength': round(old_strength, 6),
                'new_strength': round(new_strength, 6),
            }
        self.kb.conn.commit()
        if updates:
            self.kb.upsert(updates)
        return summary

    def outcome_stats(self, scope: str | None = None) -> list[dict]:
        """Tracked outcome credit-assignment statistics for admitted records."""
        if not self.kb.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='experience_outcomes'"
        ).fetchone():
            return []
        if scope is not None:
            check_scope(scope)
            rows = self.kb.conn.execute(
                'SELECT scope, record_id, injections, lift_sum, ema_lift, last_reward '
                'FROM experience_outcomes WHERE scope=? ORDER BY ema_lift DESC',
                (scope,),
            ).fetchall()
        else:
            rows = self.kb.conn.execute(
                'SELECT scope, record_id, injections, lift_sum, ema_lift, last_reward '
                'FROM experience_outcomes ORDER BY ema_lift DESC'
            ).fetchall()
        return [
            {
                'scope': r[0],
                'record_id': r[1],
                'injections': int(r[2]),
                'mean_lift': float(r[3]) / max(1, int(r[2])),
                'ema_lift': float(r[4]),
                'last_reward': float(r[5]),
            }
            for r in rows
        ]

    def prune_weak(self, *, min_injections: int = 5, min_ema_lift: float = 0.0) -> list[tuple[str, str]]:
        """Remove records with at least ``min_injections`` observations whose ``ema_lift < min_ema_lift``."""
        if self._readonly:
            raise ValueError('cannot prune a readonly ScopedExperienceKB')
        rows = self.kb.conn.execute(
            'SELECT scope, record_id FROM experience_outcomes WHERE injections >= ? AND ema_lift < ?',
            (int(min_injections), float(min_ema_lift)),
        ).fetchall()
        if not rows:
            return []
        by_scope: dict[str, list[int]] = {}
        removed: list[tuple[str, str]] = []
        for scope, rid in rows:
            by_scope.setdefault(scope, []).append(record_key(scope, rid))
            removed.append((scope, rid))
        for scope, ids in by_scope.items():
            self.kb.delete(ids, scope=scope)
        self.kb.conn.executemany(
            'DELETE FROM experience_outcomes WHERE scope=? AND record_id=?',
            removed,
        )
        self.kb.conn.executemany(
            'DELETE FROM experience_drift WHERE scope=? AND record_id=?',
            removed,
        )
        self.kb.conn.commit()
        return removed

    # ---- Direct KB entry payload updates & drift tracking (EM reproduction targets) ---------------------------------
    def update_entry_payload(
        self,
        scope: str,
        record_id: str,
        *,
        new_value: torch.Tensor | None = None,
        new_key: torch.Tensor | None = None,
        new_compacted_state_id: str | None = None,
        mark_drifted: bool = True,
    ) -> dict[str, float]:
        """Update an admitted KB entry's on-disk value/key/Neuralese state and record its drift for reproduction."""
        if self._readonly:
            raise ValueError('cannot update entry payload on a readonly ScopedExperienceKB')
        scope = check_scope(scope)
        rid = str(record_id)
        int_id = record_key(scope, rid)
        fetched = self.kb.fetch_records([int_id], scope=scope)
        if not fetched:
            raise KeyError(f'record ({scope!r}, {rid!r}) not found')
        rec = fetched[0]
        old_val = rec.value.detach().float().cpu()
        old_key = rec.key.detach().float().cpu()
        val = old_val if new_value is None else new_value.detach().float().cpu()
        key = old_key if new_key is None else F.normalize(new_key.detach().float().cpu(), dim=-1)
        if val.shape != (self.value_dim,) or not torch.isfinite(val).all():
            raise ValueError(f'new_value must be finite shape ({self.value_dim},), got {tuple(val.shape)}')
        if key.shape != (self.key_dim,) or not torch.isfinite(key).all():
            raise ValueError(f'new_key must be finite shape ({self.key_dim},), got {tuple(key.shape)}')
        meta = dict(rec.metadata)
        if new_compacted_state_id is not None:
            inner = dict(meta.get(META) or {})
            inner['compacted_state_id'] = new_compacted_state_id
            meta[META] = inner
        self.kb.upsert([Record(rec.id, key, val, scope=rec.scope, metadata=meta)])
        drift_norm = float(torch.linalg.vector_norm(val - old_val).item() + torch.linalg.vector_norm(key - old_key).item())
        row = self.kb.conn.execute(
            'SELECT drift_step, reproduced_step FROM experience_drift WHERE scope=? AND record_id=?',
            (scope, rid),
        ).fetchone()
        prev_drift = 0 if row is None else int(row[0])
        prev_repro = 0 if row is None else int(row[1])
        next_drift = prev_drift + (1 if mark_drifted else 0)
        self.kb.conn.execute(
            'INSERT OR REPLACE INTO experience_drift '
            '(scope, record_id, drift_step, reproduced_step, drift_norm) VALUES (?, ?, ?, ?, ?)',
            (scope, rid, next_drift, prev_repro, drift_norm),
        )
        self.kb.conn.commit()
        return {'drift_step': float(next_drift), 'reproduced_step': float(prev_repro), 'drift_norm': drift_norm}

    def iter_drifted_entries(self, *, scope: str | None = None, only_unreproduced: bool = True) -> list[dict]:
        """Return KB entries whose learned encodings drifted past the encoder's last reproduction pass."""
        if not self.kb.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='experience_drift'"
        ).fetchone():
            return []
        where = ['drift_step > reproduced_step'] if only_unreproduced else ['drift_step > 0']
        params: list = []
        if scope is not None:
            where.append('scope = ?')
            params.append(check_scope(scope))
        rows = self.kb.conn.execute(
            f'SELECT scope, record_id, drift_step, reproduced_step, drift_norm '
            f'FROM experience_drift WHERE {" AND ".join(where)} ORDER BY drift_step DESC, drift_norm DESC',
            tuple(params),
        ).fetchall()
        out: list[dict] = []
        for sc, rid, d_step, r_step, d_norm in rows:
            fetched = self.kb.fetch_records([record_key(sc, rid)], scope=sc)
            if not fetched:
                continue
            rec = fetched[0]
            meta = rec.metadata.get(META) or {}
            out.append({
                'scope': sc,
                'record_id': rid,
                'text': meta.get('text', ''),
                'kind': meta.get('kind', 'success'),
                'strength': float(rec.metadata.get(STRENGTH, 1.0)),
                'key': rec.key.detach().float().cpu(),
                'value': rec.value.detach().float().cpu(),
                'compacted_state_id': meta.get('compacted_state_id'),
                'drift_step': int(d_step),
                'reproduced_step': int(r_step),
                'drift_norm': float(d_norm),
            })
        return out

    def mark_reproduced(self, pairs: list[tuple[str, str]]) -> int:
        """Advance ``reproduced_step = drift_step`` after the hindsight encoder reproduces drifted entries."""
        if self._readonly:
            raise ValueError('cannot mark reproduced on a readonly ScopedExperienceKB')
        updated = 0
        for scope, rid in pairs:
            cur = self.kb.conn.execute(
                'UPDATE experience_drift SET reproduced_step = drift_step WHERE scope=? AND record_id=?',
                (check_scope(scope), str(rid)),
            )
            updated += int(cur.rowcount or 0)
        self.kb.conn.commit()
        return updated
