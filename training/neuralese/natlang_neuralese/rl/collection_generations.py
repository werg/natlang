"""A collection resumes across (re)starts that change its transport: endpoints, worker counts, even the controller code.

The collection binding (``collection-binding.json``) pins what makes the rollouts the policy's own: cases, policy,
sampling and, by default, the collector/actor code that produced them. It does not pin *where* they were sampled:
endpoints and worker counts are transport, and each endpoint's policy identity is authenticated by ``actor_receipts``
at every (re)start. Every start of ``collect`` is one *generation*: it appends its actual endpoints, slots, actor
receipts and code changes to ``generations.jsonl`` (append-only) and journals under new per-generation file names
(``workers/journal-g2-0.jsonl``), so nothing a previous generation wrote is overwritten. Entries already attempted in
ANY journal (every generation, student and teacher phase) are never dispatched again; the pending ones are split over
the current slots. An entry interrupted mid-run has a ``start`` but no ``finish`` row: it is pending, and its retained
partial job (keyed by the entry, not by the worker) lets the collector replay its journaled responses.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..common.jsonio import read_jsonl, write_json_durable

TRANSPORT_KEYS = ('endpoints', 'remote_workers')
CODE_KEYS = ('runtime', 'collector_sha256')
HISTORY = 'generations.jsonl'
_JOURNAL = re.compile(r'journal-(?:g(?P<gen>\d+)-)?(?P<phase>teacher-)?(?P<index>\d+)\.jsonl$')


def comparable_binding(binding, *, code=False):
    """``binding`` without its transport (and, with ``code``, without the code identity that may change on resume)."""
    drop = TRANSPORT_KEYS + (CODE_KEYS if code else ())
    out = {key: value for key, value in binding.items() if key not in drop}
    actor = out.get('expected_native_actor')
    if code and isinstance(actor, dict) and 'python_runtime' in actor:
        out['expected_native_actor'] = {key: value for key, value in actor.items() if key != 'python_runtime'}
    return out


def code_changes(recorded, current):
    """{field: [recorded, current]} of the code-identity fields that differ (digest of the large ``runtime`` map)."""
    changes = {}
    for key in CODE_KEYS:
        if recorded.get(key) != current.get(key):
            changes[key] = [_digest(recorded.get(key)), _digest(current.get(key))]
    old, new = ((binding.get('expected_native_actor') or {}).get('python_runtime') for binding in (recorded, current))
    if old != new:
        changes['actor_python_runtime'] = [_digest(old), _digest(new)]
    return changes


def _digest(value):
    if isinstance(value, dict):
        from ..common.hashing import canonical_json_sha256_hex
        return canonical_json_sha256_hex(value)
    return value


def resume_differences(recorded, current, *, allow_code_change=True):
    """(differences, code_changes): the strict differences that forbid resuming ``recorded`` as ``current`` (transport
    never counts; code identity only when ``allow_code_change`` is off), and the code-identity changes a resume accepts."""
    from .production import _binding_differences
    strict = _binding_differences(comparable_binding(recorded, code=allow_code_change),
                                  comparable_binding(current, code=allow_code_change))
    return strict, (code_changes(recorded, current) if allow_code_change else {})


class Generations:
    """The journals, queues and start history of one collection directory (``out``)."""

    def __init__(self, out):
        self.out = Path(out)
        self.directory = self.out / 'workers'
        self.history = self.out / HISTORY

    def journals(self):
        """Every worker journal of every generation, oldest first (legacy ``journal-N`` files are generation 0)."""
        found = []
        for path in self.directory.glob('journal-*.jsonl') if self.directory.is_dir() else ():
            match = _JOURNAL.fullmatch(path.name)
            if match:
                found.append(((int(match['gen'] or 0), bool(match['phase']), int(match['index'])), path))
        return [path for _, path in sorted(found)]

    def rows(self, path):
        return read_jsonl(path) if Path(path).exists() else []

    def attempted(self):
        """Keys with a ``finish`` row in any journal: they are never dispatched again (a retry is a reviewed new pass)."""
        return {row['key'] for path in self.journals() for row in self.rows(path) if row.get('event') == 'finish'}

    def started(self):
        return {row['key'] for path in self.journals() for row in self.rows(path) if row.get('event') == 'start'}

    def finished_endpoints(self):
        """{key: endpoint that finished it}: each journal's paired queue file recorded the endpoint it was dispatched to."""
        found = {}
        for path in self.journals():
            queue = path.with_name(path.name.replace('journal-', 'queue-', 1))
            sent = {entry['key']: entry.get('rl_endpoint') for entry in self.rows(queue)}
            for row in self.rows(path):
                if row.get('event') == 'finish' and sent.get(row['key']) is not None:
                    found[row['key']] = sent[row['key']]
        return found

    def generation(self):
        """The number of this start: one past every recorded start (0 is the legacy, unrecorded generation)."""
        previous = [row['generation'] for row in self.rows(self.history) if row.get('event') == 'start']
        return max(previous, default=0) + 1

    def record(self, event, **fields):
        self.out.mkdir(parents=True, exist_ok=True)
        with self.history.open('a') as stream:
            stream.write(json.dumps({'schema': 'natlang.rl-collection-generation/1', 'event': event,
                                     'time': time.time(), **fields}, sort_keys=True) + '\n')
            stream.flush()
            import os
            os.fsync(stream.fileno())

    def merge(self):
        """The round's ``journal.jsonl`` text: every generation's journal, oldest first."""
        return ''.join(path.read_text() for path in self.journals())


def dispatch(entries, *, directory, generation, phase, endpoints, split, run_worker, attempted):
    """Run the pending ``entries`` of one phase ('' student, 'teacher-') over one worker slot per ``endpoints`` item with
    ``run_worker(index, name, queue, journal)`` (the collector's exit code). Writes this generation's queue files; a slot
    with nothing to do starts no collector. Returns (results, pending count)."""
    directory = Path(directory)
    pending = [entry for entry in entries if entry['key'] not in attempted]
    shares = split(pending)
    directory.mkdir(parents=True, exist_ok=True)

    def worker(index):
        selected = shares[index]
        if not selected:
            return None
        endpoint = endpoints[index]
        name = f'g{generation}-{phase}{index}'
        queue, journal = directory / f'queue-{name}.jsonl', directory / f'journal-{name}.jsonl'
        for entry in selected:
            entry['rl_endpoint'] = endpoint
        queue.write_text(''.join(json.dumps(entry) + '\n' for entry in selected))
        return {'worker': index, 'generation': generation, 'returncode': run_worker(index, name, queue, journal),
                'journal': str(journal)}

    with ThreadPoolExecutor(max_workers=len(shares)) as pool:
        results = [result for result in pool.map(worker, range(len(shares))) if result is not None]
    return results, len(pending)


def persist_teacher_plan(path, student_keys, teachers):
    """The JOLT+ teacher plan is a model call (planTeacherAttempts): a restart must not re-plan and orphan started keys."""
    write_json_durable(Path(path), {'schema': 'natlang.rl-teacher-plan/1', 'student_keys': sorted(student_keys),
                                    'teachers': teachers})


def load_teacher_plan(path, student_keys):
    path = Path(path)
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    if saved.get('schema') != 'natlang.rl-teacher-plan/1' or saved.get('student_keys') != sorted(student_keys):
        return None
    return saved['teachers']
