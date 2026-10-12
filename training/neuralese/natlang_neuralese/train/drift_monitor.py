"""The drift monitor and guard shared by the Neuralese stages (the direct stage and the end-to-end stage).

One implementation (owner rule: one implementation per concept) of the policy in plans/neuralese/LINE_PARITY.md
(coordinator decision 2026-10-11): held response KL to the base model on ``--drift-rows`` monitor rows and greedy
agreement, at a cadence; past ``--drift-warn`` (or below ``--drift-agreement-warn``) the behaviour gate runs and is
REPORTED (``gate_below_floor`` when its pass falls below its reference minus ``max_gate_drop``); it never stops a stage
(owner decision 2026-10-12, plans/neuralese/DECISIONS.md: the 48-probe gate is report-only; qualification is natlang task
success, eval/task_cadence.py, plus drift). Response KL past ``--drift-stop`` stops at once (a catastrophic shift).
Crisp by design: it is an exact verifier of what the stage produced, not a policy.

The reference is either the quantization component's teacher held rows, or (``--monitor`` with ``data``) the base
model's own answers and top-k on the monitor prompts, captured before any update and kept in the run directory.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch


def add_drift_arguments(parser, *, drift_rows: int = 16) -> None:
    """The drift monitor's options (same names and meaning in every stage that runs it)."""
    parser.add_argument('--monitor', help='JSON (or a path to it) of the monitors without a quantization component: '
                        '{"gate": <quantization.gate schema>, "data": {"path", "sha256"}}; "data" is JSONL of chat '
                        'prompts whose base-model answers and top-k are captured at step 0 as the drift reference')
    parser.add_argument('--drift-rows', type=int, default=drift_rows, help='monitor rows of the held KL; 0: off')
    parser.add_argument('--drift-gate-every', type=int, default=300,
                        help='while the guard warns, run the behaviour gate at most every this many updates (KL '
                             'still every --drift-every); always at the stage end')
    parser.add_argument('--drift-gate-rise', type=float, default=0.05,
                        help='run the gate at once, whatever --drift-gate-every, when response KL has risen by this '
                             'much since the last gate')
    parser.add_argument('--gate-response-kl', type=float, default=None,
                        help="the behaviour gate's KL bound for this stage, on response positions (overrides the "
                             "gate's max_held_kl and held_kl_bound); response KL measures change from base, the gate "
                             'pass rate and held CE measure damage')
    parser.add_argument('--drift-every', type=int, default=0,
                        help='also run the drift monitor (held KL on --drift-rows, its guard and the gate at a warn) '
                             'every this many updates between evals, logged to drift.jsonl; 0: only at evals')
    parser.add_argument('--drift-warn', type=float, default=.05)
    parser.add_argument('--drift-stop', type=float, default=.5)
    parser.add_argument('--drift-agreement-warn', type=float, default=.97)
    parser.add_argument('--drift-agreement-stop', type=float, default=0.)
    parser.add_argument('--drift-gate', action=argparse.BooleanOptionalAction, default=True,
                        help='run the behaviour gate at an eval whose drift is past warn and report it (report-only '
                             'since 2026-10-12: a gate regression is logged as gate_below_floor, never a stop)')
    parser.add_argument('--gate-held-rows', type=int, default=48)


def parse_monitor(value):
    """``--monitor``: inline JSON or a path to it (None when absent)."""
    if not value:
        return None
    return json.loads(Path(value).read_text() if not value.lstrip().startswith('{') else value)


def monitor_reference(monitor, out: Path, hf, tokenizer, render, *, root):
    """The monitor-only drift reference of a lineage without a teacher artifact: the base model's own answers and
    top-k on the monitor prompts, captured before any update and kept with the run (``monitor-reference.pt``)."""
    from ..common.hashing import sha256_file_hex
    from ..maple.generation_gate import capture_reference
    reference_path = Path(out) / 'monitor-reference.pt'
    if reference_path.is_file():
        return torch.load(reference_path, map_location='cpu', weights_only=False)
    data = Path(monitor['data']['path'])
    data = data if data.is_absolute() else Path(root) / data
    if sha256_file_hex(data) != monitor['data']['sha256']:
        raise ValueError('monitor prompts changed since the recipe pinned them: ' + str(data))
    prompts = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    held = capture_reference(hf, tokenizer, prompts, render, eos)
    torch.save(held, reference_path)
    print(json.dumps({'event': 'monitor_reference_captured', 'prompts': len(prompts)}), flush=True)
    return held


def setup_monitor(monitor, out: Path, hf, tokenizer, *, behaviour=None, gate_held=None, gate_response_kl=None,
                  root):
    """(behaviour, gate_held) after ``--monitor``: the reference captured when no teacher holds one, the monitor's
    gate when no quantization component declares one, and the stage's response-KL bound on the gate."""
    from ..maple.distill_data import render
    from .quantization import Behaviour
    if monitor is not None:
        if gate_held is None and monitor.get('data'):
            gate_held = monitor_reference(monitor, out, hf, tokenizer, render, root=root)
        if behaviour is None and monitor.get('gate'):
            behaviour = Behaviour(monitor['gate'], tokenizer, gate_held, root=root)
    if behaviour is not None and gate_response_kl is not None:
        # The stage's declared KL bound (coordinator decision 2026-10-11): on response positions, at the hard stop.
        behaviour.config = {**behaviour.config, 'max_held_kl': gate_response_kl, 'held_kl_bound': 'response'}
    return behaviour, gate_held


def drift_level(drift, a) -> str:
    """The drift guard's level on a ``behaviour_drift`` report (``text_warmup.behaviour_drift_level``)."""
    from .text_warmup import behaviour_drift_level
    return behaviour_drift_level(drift, a)


class DriftMonitor:
    """The monitor of one stage run. ``a``: the parsed options (``add_drift_arguments``); ``bf16``: a callable
    returning the context under which the monitor measures (the BF16 point of a quantization component, else
    nothing). ``last_gate``/``last_gate_kl`` are run state, saved with the full state (``state``/``restore``)."""

    def __init__(self, hf, tokenizer, gate_held, behaviour, a, *, bf16=None):
        self.hf, self.tokenizer, self.gate_held, self.behaviour, self.a = hf, tokenizer, gate_held, behaviour, a
        self.bf16 = bf16 or contextlib.nullcontext
        self.last_gate = self.last_gate_kl = None
        self.gate_by_step = {}  # the full gate already run at a step (at a warn), reused as the final gate

    @property
    def active(self) -> bool:
        return self.gate_held is not None and bool(self.a.drift_rows)

    def state(self) -> dict:
        return {'last_gate': self.last_gate, 'last_gate_kl': self.last_gate_kl}

    def restore(self, state) -> None:
        self.last_gate, self.last_gate_kl = (state or {}).get('last_gate'), (state or {}).get('last_gate_kl')

    def check(self, step: int, end: int) -> dict:
        """Held KL on --drift-rows, its guard level, and at a warn the behaviour gate (reported; ``gate_below_floor``
        below its reference minus max_gate_drop) -- at most every --drift-gate-every updates, at once when response KL rose by
        --drift-gate-rise since the last gate, and at the stage end (``step >= end``)."""
        from ..maple.generation_gate import held_kl
        a = self.a
        with torch.no_grad(), self.bf16():
            drift = held_kl(self.hf, self.gate_held, a.drift_rows, 0.25)
        drift['guard'] = drift_level(drift, a)
        due = (self.last_gate is None or step - self.last_gate >= a.drift_gate_every or step >= end or
               (self.last_gate_kl is not None and drift['held_response_kl'] - self.last_gate_kl >= a.drift_gate_rise))
        if drift['guard'] == 'warn' and a.drift_gate and self.behaviour is not None and not due:
            drift['gate_at_warn'] = {'deferred': True, 'last_gate': self.last_gate}
        elif drift['guard'] == 'warn' and a.drift_gate and self.behaviour is not None:
            self.last_gate, self.last_gate_kl = step, drift['held_response_kl']
            gate = self.gate_by_step[step] = self.gate()
            floor = self.behaviour.config['reference']['gate_pass'] - self.behaviour.config.get('max_gate_drop', 0.0)
            drift['gate_at_warn'] = {k: gate[k] for k in ('gate_pass', 'gate_checks', 'gate_kinds', 'gate_results')}
            drift['gate_floor'] = floor
            drift['gate_report_only'] = True  # owner decision 2026-10-12: reported, never a stop
            if gate['gate_pass'] < floor - 1e-9:
                drift['gate_below_floor'] = True
        elif drift['guard'] == 'stop':
            drift['stop_reason'] = 'response KL or agreement past the hard stop'
        return drift

    def gate(self):
        """The full behaviour gate (None without one)."""
        if self.behaviour is None:
            return None
        from ..maple.generation_gate import behaviour_report
        b = self.behaviour
        with torch.no_grad(), self.bf16():
            result = behaviour_report(self.hf, self.tokenizer, b.probes, b.render, b.eos, self.gate_held,
                                      self.a.gate_held_rows, b.config.get('prompt_weight', 0.25))
        result['passed'] = b.passed(result)
        return result

    def final_gate(self, step: int):
        return self.gate_by_step.get(step) or self.gate()
