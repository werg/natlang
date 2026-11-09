"""A collection resumes across (re)starts that change its transport: endpoints, worker counts, even the controller code.

The collection binding (``collection-binding.json``) pins what makes the rollouts the policy's own: cases, policy and
sampling. It does not pin *where* they were sampled (endpoints and worker counts are transport; each endpoint's policy
identity is authenticated by ``actor_receipts`` at every (re)start) nor *which code* sampled them: production runs from
the checkout, so each start records the git commit of the controller and of every actor (``code`` in its ``start``
event, ``code_changes`` against the previous start) and never refuses on it. Every start of ``collect`` is one *generation*: it appends its actual endpoints, slots, actor
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
from threading import Lock

from ..common.jsonio import read_jsonl, write_json_durable

TRANSPORT_KEYS = ('endpoints', 'remote_workers')
# Fields of bindings written before code identity became provenance: never compared (new bindings do not carry them).
LEGACY_CODE_KEYS = ('runtime', 'collector_sha256')
HISTORY = 'generations.jsonl'
_JOURNAL = re.compile(r'journal-(?:g(?P<gen>\d+)-)?(?P<phase>teacher-)?(?P<index>\d+)\.jsonl$')


def comparable_binding(binding):
    """``binding`` without its transport and without any legacy code-identity field."""
    out = {key: value for key, value in binding.items() if key not in TRANSPORT_KEYS + LEGACY_CODE_KEYS}
    actor = out.get('expected_native_actor')
    if isinstance(actor, dict) and 'python_runtime' in actor:
        out['expected_native_actor'] = {key: value for key, value in actor.items() if key != 'python_runtime'}
    return out


def previous_code(history):
    """The ``code`` of the newest recorded start of ``history`` (rows of generations.jsonl), or None."""
    starts = [row.get('code') for row in history if row.get('event') == 'start' and row.get('code')]
    return starts[-1] if starts else None


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


class IncompleteCollection(RuntimeError):
    """Expected queue keys have no terminal ``finish`` row; the collection must be resumed, never sealed."""

    def __init__(self, phase, expected, missing):
        self.phase, self.expected, self.missing = phase, expected, list(missing)
        super().__init__(f"RL {phase or 'student'} collection incomplete: {len(self.missing)} of {expected} expected "
                         f"keys have no finish row (restart resumes them): {', '.join(self.missing[:8])}"
                         + (' ...' if len(self.missing) > 8 else ''))


def unfinished(entries, finished):
    """Keys of ``entries`` without a terminal ``finish`` row in ``finished`` (process exit codes prove nothing: a
    valid terminal failure exits nonzero, and a dead collector can exit zero)."""
    return [entry['key'] for entry in entries if entry['key'] not in finished]


def dispatch(entries, *, directory, generation, phase, endpoints, run_worker, attempted, stopped=lambda: False,
             ready=None, poll_seconds=15., feed=None, outside_busy=None):
    """Run the pending ``entries`` of one phase ('' student, 'teacher-') over one worker slot per ``endpoints`` item with
    ``run_worker(index, name, queue, journal)`` (the collector's exit code). Slots steal work: each takes the next pending
    entry when its previous one finishes, appends it to its own queue file and reruns its collector on that queue and
    journal (the collector skips keys its journal already finished). A static split left the faster Pop actor idle for
    hours while DGX slots still held most of a round. A slot that never takes an entry starts no collector; a stop
    request ends every slot after its current entry, and a slot whose collector leaves its entry unfinished stops taking work. ``ready(index)``, when given, is
    asked before each entry: a slot whose actor is not (yet) serving the round's policy waits instead of taking work, so
    an actor that joins mid-round (Pop after a dev16 evaluation) picks up the remaining entries, and an actor that died
    loses at most its current entry instead of fast-failing the queue (2026-11-09: 20 entries drained in 26 min).
    ``feed`` (rl.teacher_feed.TeacherFeed) supplies more entries while the queue runs: a planner thread polls it, and the
    entries it returns (JOLT+ teacher attempts of groups whose students all finished) join the END of the shared queue and
    are appended to ``entries``, so an idle slot takes them while other students still run. Slots then wait, instead of
    ending, while the queue is empty and the feed is open; the feed closes when every group is planned, or when nothing
    runs and nothing is queued (``outside_busy()``: workers detached by an earlier controller still running).
    Returns (results, counts): ``counts`` gives the ``initial`` pending entries of this call, how many were
    ``attempted`` (taken by a slot), how many ``finished`` (a finish row exists) and how many ``remaining`` unfinished
    (never taken, or taken by a collector that left no finish row)."""
    directory = Path(directory)
    pending = [entry for entry in entries if entry['key'] not in attempted]
    directory.mkdir(parents=True, exist_ok=True)
    remaining, lock = list(reversed(pending)), Lock()
    state = {'active': 0, 'closed': feed is None, 'error': None}

    def worker(index):
        endpoint = endpoints[index]
        name = f'g{generation}-{phase}{index}'
        queue, journal = directory / f'queue-{name}.jsonl', directory / f'journal-{name}.jsonl'
        taken, codes = [], []
        while not stopped():
            with lock:
                drained = not remaining and state['closed']
                waiting = not remaining and not state['closed']
            if drained:
                break
            if waiting:   # the feed may still queue teacher entries: stay (a slot that ends cannot take them)
                time.sleep(poll_seconds)
                continue
            if ready is not None and not ready(index):
                time.sleep(poll_seconds)
                continue
            with lock:
                if not remaining:
                    continue
                entry = remaining.pop()
                state['active'] += 1
            try:
                entry['rl_endpoint'] = endpoint
                taken.append(entry)
                queue.write_text(''.join(json.dumps(item) + '\n' for item in taken))
                codes.append(run_worker(index, name, queue, journal))
            finally:
                with lock:
                    state['active'] -= 1
            # The collector exits nonzero for an ordinary incomplete or timed-out episode (still a finish row). A
            # collector that left its entry without a finish row died or lost its endpoint: that slot stops taking
            # work so it cannot drain the shared queue; the entry stays pending for the next start.
            if not any(row.get('event') == 'finish' and row.get('key') == entry['key']
                       for row in (read_jsonl(journal) if journal.exists() else ())):
                break
        if not taken:
            return None
        return {'worker': index, 'generation': generation, 'entries': len(taken),
                'returncode': next((code for code in codes if code), 0), 'journal': str(journal)}

    def planner():
        try:
            while not stopped():
                with lock:
                    idle = not remaining
                    quiet = idle and state['active'] == 0 and not (outside_busy is not None and outside_busy())
                # A slot would idle (or nothing runs): plan what has finished now instead of waiting for a full batch.
                new = feed.poll(flush=idle)
                if new:
                    with lock:
                        remaining[0:0] = list(reversed(new))   # the end of the queue (taken from the right)
                        entries.extend(new)
                        pending.extend(new)
                    continue
                if feed.complete() or quiet:
                    break
                time.sleep(min(poll_seconds, 5.))
        except BaseException as error:   # surfaced by dispatch after the slots end
            state['error'] = error
        finally:
            with lock:
                state['closed'] = True

    import threading
    thread = threading.Thread(target=planner, name='teacher-feed', daemon=True) if feed is not None else None
    if thread is not None:
        thread.start()
    try:
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            results = [result for result in pool.map(worker, range(len(endpoints))) if result is not None]
    finally:
        if thread is not None:
            with lock:
                state['closed'] = state['closed'] or stopped()
            thread.join(timeout=5.)
    if state['error'] is not None:
        raise state['error']
    finished = {row['key'] for result in results for row in (read_jsonl(result['journal']) if Path(result['journal']).exists() else ())
                if row.get('event') == 'finish'}
    counts = {'initial': len(pending), 'attempted': sum(result['entries'] for result in results),
              'finished': sum(entry['key'] in finished for entry in pending)}
    counts['remaining'] = counts['initial'] - counts['finished']
    return results, counts


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
