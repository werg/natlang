"""JOLT+ teacher planning that overlaps the student phase (owner request 2026-11-10).

A round used to plan every teacher attempt in one planTeacherAttempts call after ALL students finished, so the machines
idled through the plan, the teacher phase tail and the learner start (w4 round 1: 128 students 8.3 h, plan 2 min, 16
teachers 1.4 h on 8 slots). The feed instead watches the student journals: as soon as every student of a group has a
finish row the group is scored, and the groups that finished since the last call are planned together
(``teacher_entries`` over just those groups) against a running budget, ``attempts`` per planned group plus whatever
earlier batches left unspent, so the round total never exceeds ``attempts x groups`` (the old budget). The new teacher
entries join the same work-stealing dispatch (``collection_generations.dispatch(feed=...)``): idle slots take them while
other students still run. Group scoring and the per-group student baselines are unchanged (``score`` over the group's
own queue entries); the plan validation and crisp fallback are ``jolt.plan_teacher_attempts``'s.

``teacher-plan.json`` (schema /2) is appended batch by batch, durably, BEFORE a batch's entries are queued: a restart
reuses the recorded batches (and their keys) exactly and only plans groups no batch covers, so a started teacher key is
never re-planned. A round whose plan was written by the old whole-round planner (schema /1) resumes through the old path.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common.jsonio import write_json_durable

SCHEMA = 'natlang.rl-teacher-plan/2'
DEFAULT_BATCH = 4


def score_student_groups(out, members, journal_rows, *, score, bonus=None):
    """Scored rollout rows of exactly ``members`` (the students of finished groups), in a scratch run directory: the
    round's own score.json / rewards.jsonl are never touched and unfinished students are never scored.
    ``journal_rows``: rows of the round's worker journals."""
    import shutil
    out = Path(out)
    scratch = out / 'teacher-plan-scoring'
    scratch.mkdir(exist_ok=True)
    for name in ('packet.jsonl', 'preparation.json'):
        shutil.copyfile(out / name, scratch / name)
    keys = {entry['key'] for entry in members}
    (scratch / 'queue.jsonl').write_text(''.join(json.dumps(entry) + '\n' for entry in members))
    (scratch / 'journal.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in journal_rows if row.get('key') in keys))
    score(scratch, scratch / 'rollouts.jsonl', neuralese_bonus=bonus)
    return [json.loads(line) for line in (scratch / 'rollouts.jsonl').read_text().splitlines()]


class TeacherFeed:
    """Incremental teacher plan of one collection. ``plan_batch(members, rows, carry)`` -> ``teacher_entries`` result
    for the student entries ``members`` of finished groups; ``score_rows(members)`` -> their scored rollout rows;
    ``finished()`` -> keys with a finish row; ``bind(entries)`` applies the round's prompt/context/endpoint binding."""

    def __init__(self, *, path, students, attempts, finished, score_rows, plan_batch, bind, batch=DEFAULT_BATCH,
                 on_plan=None):
        self.path, self.attempts, self.batch = Path(path), attempts, max(1, int(batch))
        self.finished, self.score_rows, self.plan_batch, self.bind, self.on_plan = finished, score_rows, plan_batch, bind, on_plan
        self.groups = {}
        for entry in students:
            self.groups.setdefault(entry['group'], []).append(entry)
        self.student_keys = sorted(entry['key'] for entry in students)
        self.batches = []
        self.teachers = {'entries': [], 'refusal': None, 'plan': None}
        if self.path.is_file():
            saved = json.loads(self.path.read_text())
            if saved.get('schema') == SCHEMA and saved.get('student_keys') == self.student_keys:
                self.batches = saved['batches']
                for batch in self.batches:
                    self.teachers['entries'].extend(batch['entries'])
                self._summarize()

    @staticmethod
    def is_legacy_plan(path, student_keys):
        """True when ``path`` holds a whole-round (schema /1) plan of exactly these students: the old path resumes it."""
        path = Path(path)
        if not path.is_file():
            return False
        saved = json.loads(path.read_text())
        return saved.get('schema') == 'natlang.rl-teacher-plan/1' and saved.get('student_keys') == sorted(student_keys)

    def planned_groups(self):
        return {group for batch in self.batches for group in batch['groups']}

    def complete(self):
        return self.planned_groups() >= set(self.groups)

    def carry(self):
        """Budget the recorded batches left unspent (each batch's budget already includes the carry it received)."""
        carry = 0
        for batch in self.batches:
            if batch.get('budget') is not None:
                carry = max(0, batch['budget'] - batch['planned'])
        return carry

    def ready_groups(self, done=None):
        done = self.finished() if done is None else done
        planned = self.planned_groups()
        return [group for group in sorted(self.groups) if group not in planned
                and all(entry['key'] in done for entry in self.groups[group])]

    def _summarize(self):
        records = [batch.get('plan') or {} for batch in self.batches]
        failures = [record['nl_failure'] for record in records if record.get('nl_failure')]
        implementations = {record.get('implementation') for record in records}
        self.teachers['refusal'] = next((batch['refusal'] for batch in self.batches if batch.get('refusal')), None)
        self.teachers['plan'] = {'implementation': implementations.pop() if len(implementations) == 1 else 'mixed',
                                 'batches': len(self.batches), **({'nl_failures': failures} if failures else {})}

    def poll(self, *, flush=False):
        """Plan the finished-but-unplanned groups when ``batch`` of them have accumulated, or at once when ``flush`` (a
        slot would idle or nothing else runs). Returns the new teacher entries (already persisted, already bound)."""
        ready = self.ready_groups()
        if not ready or (len(ready) < self.batch and not flush):
            return []
        members = [entry for group in ready for entry in self.groups[group]]
        carry = self.carry()
        result = self.plan_batch(members, self.score_rows(members), carry)
        entries = list(result['entries'])
        if entries:
            self.bind(entries)
        accounting = result.get('accounting') or {}
        batch = {'groups': ready, 'budget': accounting.get('budget'), 'planned': accounting.get('planned', len(entries)),
                 'carry': carry, 'plan': result.get('plan'), 'refusal': result.get('refusal'), 'entries': entries}
        self.batches.append(batch)
        # Durable before any of these entries is queued: a restart reuses exactly this plan.
        write_json_durable(self.path, {'schema': SCHEMA, 'student_keys': self.student_keys, 'batches': self.batches})
        self.teachers['entries'].extend(entries)
        self._summarize()
        if self.on_plan is not None:
            self.on_plan(batch)
        return entries
