"""Held-family task outcomes: the ``bank_metalearning`` promotion gate (owner, 2026-11-03).

A bank snapshot promotes when, adapted per held family, it beats the current published bank (initially the
un-meta-trained bank) on exact-oracle task success on the family's development tasks; confirmation tasks are scored
once afterwards. Teacher-forced loss (`meta_bank.BankMetaLearner.evaluate`) stays the cheap steering signal. The
parent and plain per-program LoRA at several pass budgets are reported references, not gates.

Held families are RL task families (train-split rows of the RL task source, each with its exact oracle) that never
enter meta-training. Per family the tasks split into support, development and confirmation tasks. The adaptation
evidence is the parent's own oracle-verified successful rollouts on the support tasks (collected once per parent,
shared by every arm; `meta_bank.rollout_rows`). Each arm fits on that evidence, its adapter becomes the served
model's default adapter, and the development (or confirmation) tasks run through the production collector
(`rl.production.collect`) against the in-process native server, scored by the shared task scorer.
A family whose parent solves no support task has no evidence: every bank arm runs unadapted there.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path

from ...common.jsonio import write_json_durable
from . import LearnedOptimizerError

PLAN_SCHEMA = 'natlang.bank-task-families/1'
OUTCOME_SCHEMA = 'natlang.bank-task-outcomes/1'
DEFAULT_REFERENCE_PASSES = (1, 4, 24)


def _rank(seed, value):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def read_cases(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def task_families(cases, *, protected, exclude=(), support=4, dev=4, confirmation=4, max_families=16, seed=0):
    """Held task families of an RL task source: train rows only, protected-related rows dropped, families named in
    ``exclude`` (meta-training families) skipped; each family's tasks ordered by a seeded hash and cut into support,
    development and confirmation tasks. Families with fewer tasks are left out."""
    path = Path(cases)
    by_family = {}
    for row in read_cases(path):
        if row.get('split') != 'train' or not row.get('family'):
            continue
        if protected is not None and protected.related({'id': row['id'], 'source_groups': row.get('source_groups') or [],
                                                        'source_ids': row.get('source_ids') or []}):
            continue
        by_family.setdefault(row['family'], []).append(row['id'])
    need = support + dev + confirmation
    chosen = {}
    for family in sorted(by_family, key=lambda name: _rank(seed, name)):
        if family in exclude or len(by_family[family]) < need:
            continue
        ids = sorted(by_family[family], key=lambda value: _rank(seed, value))
        chosen[family] = {'support': ids[:support], 'dev': ids[support:support + dev],
                          'confirmation': ids[support + dev:need]}
        if len(chosen) >= max_families:
            break
    return {'schema': PLAN_SCHEMA, 'cases': {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
            'seed': seed, 'tasks': {'support': support, 'dev': dev, 'confirmation': confirmation}, 'families': chosen}


class TaskOutcomes:
    """Collections of one stage's held task families on the learner's engine (`collect` is injectable for tests)."""

    def __init__(self, learner, plan, out, *, parent_sha256, protected, samples=4, workers=4, case_seconds=600,
                 max_model_requests=64, max_turns=64, max_tokens=16384, reference_passes=DEFAULT_REFERENCE_PASSES,
                 collect=None, endpoint=None):
        self.learner, self.engine, self.plan, self.out = learner, learner.engine, plan, Path(out)
        self.parent, self.protected = parent_sha256, protected
        self.samples, self.workers, self.case_seconds = samples, workers, case_seconds
        self.max_model_requests, self.max_turns, self.max_tokens = max_model_requests, max_turns, max_tokens
        self.reference_passes = tuple(reference_passes)
        if hashlib.sha256(Path(plan['cases']['path']).read_bytes()).hexdigest() != plan['cases']['sha256']:
            raise LearnedOptimizerError('the RL task source changed after the held task families were chosen')
        self.rows = {row['id']: row for row in read_cases(plan['cases']['path'])}
        self.collect = collect
        self.endpoint = endpoint
        self.server = None

    # -- serving ------------------------------------------------------------------------------------------------
    def _url(self):
        if self.endpoint is None:
            from ...serve import configure_native_prefix_cache
            from ...serve.http import serve
            configure_native_prefix_cache(self.engine, 2.0)
            self.engine.start()
            self.server = serve(self.engine, '127.0.0.1', 0)
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            host, port = self.server.server_address[:2]
            self.endpoint = f'http://{host}:{port}'
        return self.endpoint

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.engine.stop()
            self.server = None

    def run(self, name, ids, adapter=None, binding=None):
        """Collect ``samples`` attempts of each task in ``ids`` with ``adapter`` (an adapter id or None: the parent)
        as the served model's default; resumes a completed or partial collection in ``out/name``. Returns the
        success counts."""
        from ...rl.production import collect
        collect = self.collect or collect
        directory = self.out / name
        directory.mkdir(parents=True, exist_ok=True)
        packet = directory.parent / f'{directory.name}-cases.jsonl'
        content = ''.join(json.dumps(self.rows[task], ensure_ascii=False) + '\n' for task in ids)
        if packet.exists() and packet.read_text() != content:
            raise LearnedOptimizerError(f'{packet} holds other tasks')
        packet.write_text(content)
        engine = self.engine
        previous = (getattr(engine, 'native_default_adapters', None), dict(getattr(engine, 'rl_policy_binding', {}) or {}))
        engine.native_default_adapters = [{'id': adapter, 'scale': 1.}] if adapter else None
        if hasattr(engine, 'rl_policy_binding'):
            engine.rl_policy_binding = {**previous[1], 'bank_adaptation': binding} if binding else dict(previous[1])
        try:
            report = collect(cases=packet, out=directory, endpoints=[self._url()], policy_id=self.parent, rows=len(ids),
                             group=self.samples, workers=self.workers, root_mode='student', prompt_rendering='full',
                             case_seconds=self.case_seconds, max_model_requests=self.max_model_requests,
                             max_turns=self.max_turns, neuralese_use_bonus=0)
        finally:
            engine.native_default_adapters = previous[0]
            if hasattr(engine, 'rl_policy_binding'):
                engine.rl_policy_binding = previous[1]
        per_task = {task: 0 for task in ids}
        path = directory / 'rollouts.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                rollout = json.loads(line)
                if (rollout.get('reward') or {}).get('quality') == 1.0 and rollout.get('task_id') in per_task:
                    per_task[rollout['task_id']] += 1
        return {'successes': int(report['successes']), 'scored': int(report['scored']), 'per_task': per_task,
                'adapter': adapter, 'binding': binding}

    # -- adaptation ---------------------------------------------------------------------------------------------
    def support_rows(self, family):
        """The parent's oracle-verified successful rollouts on the family's support tasks (once per parent)."""
        from .meta_bank import rollout_rows
        result = self.run(f'{self.parent[:16]}/{family}/support', self.plan['families'][family]['support'])
        rows, counts = rollout_rows(self.engine, [self.out / self.parent[:16] / family / 'support' / 'rollouts.jsonl'],
                                    protected=self.protected, max_tokens=self.max_tokens, policy=self.parent)
        return rows, {**counts, 'rows': len(rows), 'successes': result['successes'], 'scored': result['scored']}

    def bank_adapter(self, which, rows):
        """Fit bank ``which`` on ``rows`` with the learner's inner step and store the composed adapter."""
        learner = self.learner
        bank = learner.bank_of(which)
        fitted = learner.fit(rows, which)
        slots = list(range(len(bank.components)))
        cores = bank.composed_cores(slots, fitted['coefficients'])
        record = bank.compose_adapter(self.engine, cores, provenance={'bank_adaptation': which,
                                                                       'support': [row['graph']['id'] for row in rows]})
        return record.id, {'bank': bank.snapshot, 'which': which, 'support_loss': fitted['history'][-1]}

    def lora_adapter(self, rows, passes):
        learner = self.learner
        leaves, history = learner.fit_lora(rows, passes)
        base = self.engine.store.get_native_tensor_artifact(learner.lora).artifact
        record = self.engine.store.put_native_tensor_artifact(replace(
            base, tensors={key: value.detach().cpu().contiguous() for key, value in leaves.items()}))
        return record.id, {'lora_passes': passes, 'support_loss': history[-1]}

    def evaluate(self, split='dev', *, references=True):
        """Every held task family on ``split`` tasks: the candidate (live) and published banks, and with
        ``references`` the parent and plain LoRA at each reference pass budget. Resumable per collection."""
        families, out = {}, {'schema': OUTCOME_SCHEMA, 'split': split, 'parent_sha256': self.parent,
                             'candidate_step': self.learner.state['step'], 'samples': self.samples}
        for family, tasks in sorted(self.plan['families'].items()):
            rows, support = self.support_rows(family)
            record = {'support': support, 'arms': {}}
            ids = tasks[split]
            arms = [('candidate', 'meta'), ('published', 'published')]
            if references:
                arms = [('parent', None), *arms, *[(f'lora@{n}', n) for n in self.reference_passes]]
            for arm, spec in arms:
                if not rows or arm == 'parent':
                    # Without adaptation evidence every arm is the parent itself: one collection serves them all.
                    if 'parent' not in record['arms']:
                        record['arms']['parent'] = self.run(f'{self.parent[:16]}/{family}/{split}/parent', ids)
                    record['arms'][arm] = {**record['arms']['parent'], 'unadapted': not rows}
                    continue
                if arm.startswith('lora@'):
                    adapter, binding = self.lora_adapter(rows, spec)
                else:
                    adapter, binding = self.bank_adapter(spec, rows)
                # The directory names the exact adapter: a changed bank or fit is a new collection, an unchanged
                # one (the published bank, the references) resumes or reuses its completed collection.
                key = hashlib.sha256(adapter.encode()).hexdigest()[:16]
                record['arms'][arm] = self.run(f'{self.parent[:16]}/{family}/{split}/{arm}-{key}', ids, adapter,
                                               {'arm': arm, **binding})
            families[family] = record
        out['families'] = families
        out['totals'] = {arm: sum(f['arms'][arm]['successes'] for f in families.values() if arm in f['arms'])
                         for arm in sorted({a for f in families.values() for a in f['arms']})}
        out['scored'] = {arm: sum(f['arms'][arm]['scored'] for f in families.values() if arm in f['arms'])
                         for arm in out['totals']}
        out['adapted_families'] = sum(1 for f in families.values() if f['support']['rows'])
        write_json_durable(self.out / f'{split}-step{self.learner.state["step"]}.json', out)
        return out


def outcome_promotion(evaluation, *, min_families=1, margin=0):
    """The gate: the candidate's held-family task successes exceed the published bank's by more than ``margin``
    over families where the banks could adapt (at least ``min_families``)."""
    totals = evaluation.get('totals') or {}
    adapted = evaluation.get('adapted_families') or 0
    if adapted < min_families or 'candidate' not in totals or 'published' not in totals:
        return {'met': False, 'basis': 'task-outcome', 'adapted_families': adapted,
                'reason': f'{adapted} held task families had adaptation evidence; {min_families} required'}
    met = totals['candidate'] > totals['published'] + margin
    return {'met': met, 'basis': 'task-outcome', 'adapted_families': adapted, 'totals': totals,
            'scored': evaluation.get('scored'),
            'reason': ('candidate bank beats the published bank on held-family task success' if met else
                       f'candidate {totals["candidate"]} vs published {totals["published"]} successes: not better')}
