"""In-process update profiler for the trainers (py-spy cannot attach under the container's ptrace policy).

``NATLANG_PROFILE_UPDATES=a:b`` records updates a..b (the 1-based update numbers of ``train.jsonl``, absolute across a
resume) with ``torch.profiler`` and writes ``<out>/profile-<a>-<b>.json`` (the breakdown) and
``<out>/profile-<a>-<b>.trace.json.gz`` (a Chrome/Perfetto trace). Unset, every hook here is a constant no-op: the
scopes return one shared ``nullcontext`` and nothing is recorded.

Attribution of GPU (kernel) time (``analyze_trace``, from the exported trace: each kernel by its launch time to the
scopes open on the launching thread):

- trainer regions ``update/<name>`` (plan, data, forward, backward, optimizer, eval, drift, save): the outermost one
  that encloses a kernel's launching op is its phase;
- model scopes ``<phase>/<name>`` (attention, attention.sdpa, moe, moe.route, moe.experts, norm, qat, readout,
  projection), named with the active autograd phase: ``fwd`` in the forward, ``recompute`` when activation
  checkpointing replays the forward inside backward; the innermost one wins;
- optimizer scopes ``optim/<name>`` (muonsr, muonsr.newton_schulz, lionsr, adamw), also when the latents step inside
  backward;
- a backward kernel outside any scope is attributed to the forward scope of the op whose gradient it computes (the
  profiler's autograd sequence number), as ``bwd/<scope>``.

GPU idle time inside the window (wall minus the union of kernel intervals) is the host side: data, planning, Python
dispatch, synchronizations.
"""
from __future__ import annotations

import contextlib
import gzip
import json
import os
import shutil
import time
from pathlib import Path

ENABLED = False  # True only while a profiler records
_PHASE = ['fwd']
_NULL = contextlib.nullcontext()
ENV = 'NATLANG_PROFILE_UPDATES'


def scope(name: str):
    """A model scope (named with the autograd phase). No-op unless a profile is recording."""
    if not ENABLED:
        return _NULL
    import torch
    return torch.profiler.record_function(_PHASE[0] + '/' + name)


def region(name: str):
    """A fixed-name region (``update/...``, ``optim/...``). No-op unless a profile is recording."""
    if not ENABLED:
        return _NULL
    import torch
    return torch.profiler.record_function(name)


@contextlib.contextmanager
def _phase(name):
    previous = _PHASE[0]
    _PHASE[0] = name
    try:
        yield
    finally:
        _PHASE[0] = previous


def backward_phase():
    """Around ``loss.backward()``: model scopes entered there are checkpoint recomputation."""
    if not ENABLED:
        return _NULL
    return _phase('recompute')


def parse_range(value: str | None):
    """(a, b) from ``a:b`` (or ``a``); None, profiling off, for unset, empty, ``0``, ``0:0``, ``off`` or ``none``."""
    if not value or value.strip().lower() in ('0', '0:0', 'off', 'none'):
        return None
    first, _, last = value.partition(':')
    a, b = int(first), int(last or first)
    if a < 1 or b < a:
        raise ValueError(f'{ENV} must be a:b with 1 <= a <= b, got {value!r}')
    return a, b


_MODEL_PREFIXES = ('fwd/', 'recompute/')


class UpdateProfiler:
    """Records updates ``a..b`` of a training loop. Call ``begin(update)`` before each update's work,
    ``update_done(update, seconds)`` after it, and ``maybe_finish(update)`` at the end of the loop body (after its
    eval/save, which are then included); ``close(update)`` after the loop writes a partial window."""

    def __init__(self, out: Path, device=None, value: str | None = None):
        self.range = parse_range(value if value is not None else os.environ.get(ENV))
        self.out = Path(out)
        self.device = device
        self.profile = None
        self.done = False
        self.updates = {}
        self.started = None

    @property
    def active(self) -> bool:
        return self.profile is not None

    def _sync(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def begin(self, update: int) -> None:
        global ENABLED
        if self.range is None or self.done or self.profile is not None:
            return
        a, b = self.range
        if not a <= update <= b:
            return
        import torch
        from torch.profiler import ProfilerActivity, profile
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
        self._sync()
        self.profile = profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False)
        self.profile.start()
        self.first = update
        self.started = time.perf_counter()
        ENABLED = True
        print(json.dumps({'event': 'profile_started', 'update': update, 'range': list(self.range)}), flush=True)

    def update_done(self, update: int, seconds: float, **extra) -> None:
        if self.profile is not None:
            self.updates[update] = {'seconds': seconds, **extra}

    def maybe_finish(self, update: int) -> None:
        if self.profile is not None and update >= self.range[1]:
            self.close(update)

    def close(self, update: int) -> None:
        global ENABLED
        if self.profile is None:
            return
        self._sync()
        wall = time.perf_counter() - self.started
        ENABLED = False
        self.profile.stop()
        profile, self.profile, self.done = self.profile, None, True
        stem = f'profile-{self.first}-{update}'
        trace = self.out / f'{stem}.trace.json'
        try:
            profile.export_chrome_trace(str(trace))
            report = analyze_trace(json.loads(trace.read_text()), wall)
            with trace.open('rb') as source, gzip.open(str(trace) + '.gz', 'wb') as target:
                shutil.copyfileobj(source, target)
        except Exception as error:  # the breakdown must never stop a training run
            report = {'error': repr(error)}
        finally:
            trace.unlink(missing_ok=True)
        report.update(schema='natlang.update-profile/2', updates=self.updates, first=self.first, last=update,
                      requested=list(self.range), wall_seconds=wall)
        (self.out / f'{stem}.json').write_text(json.dumps(report, indent=1) + '\n')
        print(json.dumps({'event': 'profile_written', 'path': str(self.out / f'{stem}.json'),
                          'summary': report.get('summary')}), flush=True)


def _union_seconds(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total / 1e6


def analyze_trace(trace: dict, wall: float) -> dict:
    """The breakdown from the exported Chrome trace: each kernel is attributed by its launch (the runtime call that
    enqueued it) to the scopes open on the launching thread at that moment, innermost first. Event-tree parents miss
    scopes entered inside autograd hooks (the latents' in-backward optimizer step) and launches outside any aten op
    (Triton kernels). A backward kernel outside any scope goes to the forward scope of the autograd node it runs
    (matched by sequence number), as ``bwd/<scope>``."""
    events = trace.get('traceEvents', [])
    kernels = {}
    for event in events:
        if event.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and 'correlation' in event.get('args', {}):
            kernels[event['args']['correlation']] = event
    launches, spans = {}, {}
    for event in events:
        cat = event.get('cat')
        if cat in ('cuda_runtime', 'cuda_driver') and event.get('args', {}).get('correlation') in kernels:
            launches.setdefault(event['tid'], []).append(event)
        elif cat in ('cpu_op', 'user_annotation') and 'dur' in event:
            spans.setdefault(event['tid'], []).append(event)
    marks = ('update/', 'optim/') + _MODEL_PREFIXES
    forward_scope, attributed = {}, []
    for tid, items in spans.items():
        timeline = [(e['ts'], 1, e) for e in items] + [(e['ts'], 2, e) for e in launches.get(tid, [])]
        timeline.sort(key=lambda item: (item[0], item[1], -item[2].get('dur', 0)))
        stack = []
        for ts, kind, event in timeline:
            while stack and stack[-1]['ts'] + stack[-1]['dur'] < ts:
                stack.pop()
            if kind == 1:
                stack.append(event)
                sequence = event.get('args', {}).get('Sequence number')
                if sequence is not None and not event['name'].startswith('autograd::'):
                    scope = next((s['name'] for s in reversed(stack) if s['name'].startswith(_MODEL_PREFIXES)), None)
                    if scope is not None:
                        forward_scope.setdefault(sequence, scope.split('/', 1)[1])  # one forward thread
            else:
                attributed.append((list(reversed(stack)), event, tid))
    by_phase, by_category, by_kernel, kernels_by_category, intervals = {}, {}, {}, {}, []
    for event in kernels.values():
        intervals.append((event['ts'], event['ts'] + event.get('dur', 0)))
    nodes = {}
    for chain, launch, tid in attributed:
        kernel = kernels[launch['args']['correlation']]
        seconds = kernel.get('dur', 0) / 1e6
        names = [e['name'] for e in chain]
        phase = next((n for n in reversed(names) if n.startswith('update/')), 'outside')
        category = next((n for n in names if n.startswith(_MODEL_PREFIXES) or n.startswith('optim/')), None)
        if category is not None and any(n.startswith('update/') for n in names[:names.index(category)]):
            category = None  # an update region inside the scope: not this scope's work
        if category is None:
            node = next((e for e in chain if e.get('args', {}).get('Sequence number') is not None
                         and (e['name'].startswith('autograd::engine') or 'Backward' in e['name'])), None)
            if node is not None:
                args = node['args']
                scope = forward_scope.get(args['Sequence number'])
                label = node['name'].split(':')[-1].strip()
                if scope is None:
                    nodes[label] = nodes.get(label, 0) + 1
                category = 'bwd/' + scope if scope else 'bwd/unscoped:' + label
            else:
                category = 'other'
        by_phase[phase] = by_phase.get(phase, 0.0) + seconds
        by_category[category] = by_category.get(category, 0.0) + seconds
        entry = by_kernel.setdefault(kernel['name'][:160], [0, 0.0])
        entry[0] += 1
        entry[1] += seconds
        per = kernels_by_category.setdefault(category, {})
        per[kernel['name'][:100]] = per.get(kernel['name'][:100], 0.0) + seconds
    busy = _union_seconds(intervals)
    kernel_total = sum(by_category.values())

    def share(predicate):
        return sum(v for k, v in by_category.items() if predicate(k))

    summary = {
        'wall_seconds': wall, 'gpu_busy_seconds': busy, 'gpu_idle_seconds': max(0.0, wall - busy),
        'gpu_kernel_seconds': kernel_total,
        'forward': share(lambda k: k.startswith('fwd/')),
        'recompute': share(lambda k: k.startswith('recompute/')),
        'backward': share(lambda k: k.startswith('bwd/')),
        'optimizer_newton_schulz': share(lambda k: k == 'optim/muonsr.newton_schulz'),
        'optimizer_muonsr_other': share(lambda k: k == 'optim/muonsr'),
        'optimizer_other': share(lambda k: k.startswith('optim/') and not k.startswith('optim/muonsr')),
        'other': share(lambda k: not k.startswith(('fwd/', 'recompute/', 'bwd/', 'optim/'))),
    }
    groups = {}
    for key, value in by_category.items():
        base = key.split('/', 1)[-1] if key.startswith(('fwd/', 'recompute/', 'bwd/')) else key
        base = 'unscoped' if base.startswith('unscoped:') else base
        groups[base] = groups.get(base, 0.0) + value
    top = sorted(by_kernel.items(), key=lambda kv: -kv[1][1])[:40]
    return {'summary': summary,
            'by_phase_gpu_seconds': dict(sorted(by_phase.items(), key=lambda kv: -kv[1])),
            'by_scope_gpu_seconds': dict(sorted(groups.items(), key=lambda kv: -kv[1])),
            'by_category_gpu_seconds': dict(sorted(by_category.items(), key=lambda kv: -kv[1])),
            'top_kernels': [{'name': n, 'count': c, 'seconds': t} for n, (c, t) in top],
            'top_kernels_by_category': {c: dict(sorted(k.items(), key=lambda kv: -kv[1])[:6])
                                        for c, k in kernels_by_category.items()
                                        if sum(k.values()) >= 0.01 * max(kernel_total, 1e-9)},
            'unscoped_backward_nodes': dict(sorted(nodes.items(), key=lambda kv: -kv[1])[:30])}
