"""Exact evidence facts and the permitted compact evidence view (plan "What evidence analysis is", "Compact
evidence views", "Data roles and splitting").

The runtime knows these facts exactly; no model guesses them: which roots passed or failed, the first failing
invocation and binding of each failed graph, the error kind, the verified action beside the attempted one where a
demonstration exists, and per-binding failure counts. The proposer receives a narrow typed view built field by field
from a whitelist, so ``after`` bundles, supervised targets, query outcomes and protected cases cannot leak through a
convenient serializer.
"""
from __future__ import annotations

from ...common.hashing import canonical_json_sha256_hex
from . import LearnedOptimizerError

VIEW_SCHEMA = 'natlang.lopt.evidence-view/1'
ROLES = ('support', 'query', 'development', 'confirmation')
ERROR_KINDS = ('type', 'tool', 'wrong-value', 'missing-effect', 'budget-stop', 'unclassified')
_FAILED_OUTCOMES = ('rejected', 'refused', 'error', 'failed')


def item_id(kind, payload):
    return f'nev1:{kind}:' + canonical_json_sha256_hex(payload)[:32]


def attempt_facts(rollout, role):
    """One student attempt (`natlang.tune-rollout/v1`) as exact facts."""
    episode = rollout['episode']
    turns = episode.get('turns') or []
    verdict = rollout.get('verdict') or {}
    passed = rollout['reward']['quality'] == 1
    failing = None
    if not passed:
        # The oracle's recorded failure location when it names one; else the first turn whose action failed; else the
        # root's final invocation (where the wrong result was returned).
        failing = verdict.get('invocation')
        if failing is None:
            failing = next((turn['invocation_id'] for turn in turns
                            if any((call.get('outcome') in _FAILED_OUTCOMES) for call in (turn.get('calls') or []))), None)
        if failing is None and turns:
            failing = turns[-1]['invocation_id']
    kind = verdict.get('kind', 'unclassified') if not passed else None
    if kind is not None and kind not in ERROR_KINDS:
        raise LearnedOptimizerError(f'unknown error kind {kind}')
    return {'kind': 'attempt', 'role': role, 'root': rollout['root'], 'passed': passed,
            'failingInvocation': failing, 'failingBinding': rollout['bindings'].get(failing) if failing else None,
            'errorKind': kind, 'bindings': sorted({b for b in rollout['bindings'].values() if b}),
            'source': {'schema': rollout['schema'], 'sha256': canonical_json_sha256_hex(rollout)},
            'actions': [{'invocation': turn['invocation_id'], 'binding': rollout['bindings'].get(turn['invocation_id']),
                         'calls': [[call.get('name'), call.get('arguments')] for call in (turn.get('calls') or [])],
                         'observations': [call.get('result') for call in (turn.get('calls') or [])]} for turn in turns]}


def demonstration_facts(turn, role):
    """One verified teacher turn (`natlang.tune-turn/v1`)."""
    from ..rows import verify_action_evidence
    if verify_action_evidence(turn) != 'admissible':
        raise LearnedOptimizerError('only verified teacher actions are demonstrations')
    return {'kind': 'demonstration', 'role': role, 'root': turn['root'], 'binding': turn['binding'],
            'invocation': f"{turn['root']}:{turn.get('call') or 'root'}", 'teacher': turn['teacher'],
            'calls': (turn.get('reply') or {}).get('calls') or [], 'text': (turn.get('reply') or {}).get('text') or '',
            'source': {'schema': turn['schema'], 'sha256': canonical_json_sha256_hex(turn)}}


def prior_step_facts(step, role):
    """An earlier improvement step of this episode with its measured support outcome (never its query outcome)."""
    return {'kind': 'prior-step', 'role': role, 'step': step['trajectory']['step'], 'operator': step['operator']['kind'],
            'edits': [item['artifact'] for item in step['proposal']['deltas']],
            'supportGain': ((step.get('outcome') or {}).get('support') or None),
            'source': {'schema': step['schema'], 'sha256': step['id']}}


def build_view(program_contract, bundle_id, items, *, revealed_steps=()):
    """The proposer's permitted view: support items only, whitelisted fields, stable order, per-binding counts."""
    leaked = [item for item in items if item['role'] != 'support']
    if leaked:
        raise LearnedOptimizerError(f'{len(leaked)} evidence items are not support evidence; the proposer may not see them')
    allowed = {'attempt': ('kind', 'root', 'passed', 'failingInvocation', 'failingBinding', 'errorKind', 'bindings',
                           'actions', 'source'),
               'demonstration': ('kind', 'root', 'binding', 'invocation', 'calls', 'text', 'source'),
               'prior-step': ('kind', 'step', 'operator', 'edits', 'supportGain', 'source')}
    rows = []
    for item in items:
        fields = allowed.get(item['kind'])
        if fields is None:
            raise LearnedOptimizerError(f'unknown evidence kind {item["kind"]}')
        row = {field: item.get(field) for field in fields}
        row['id'] = item_id(item['kind'], row['source'])
        rows.append(row)
    rows.sort(key=lambda row: (row['kind'], row.get('root') or '', row.get('step') or 0, row['id']))
    failures, attempts = {}, {}
    for row in rows:
        if row['kind'] == 'attempt':
            for binding in row['bindings']:
                attempts[binding] = attempts.get(binding, 0) + 1
            if not row['passed'] and row['failingBinding']:
                failures[row['failingBinding']] = failures.get(row['failingBinding'], 0) + 1
    body = {'schema': VIEW_SCHEMA, 'program': program_contract, 'bundle': bundle_id, 'items': rows,
            'facts': {'roots': len({row['root'] for row in rows if row.get('root')}),
                      'passed': sum(1 for row in rows if row['kind'] == 'attempt' and row['passed']),
                      'failed': sum(1 for row in rows if row['kind'] == 'attempt' and not row['passed']),
                      'failuresByBinding': dict(sorted(failures.items())), 'attemptsByBinding': dict(sorted(attempts.items()))}}
    return {**body, 'id': 'nevv1:' + canonical_json_sha256_hex(body)}


def key_evidence_rule(view, focus, limit=6):
    """Crisp `keyEvidence` baseline: failed attempts of each focused binding first, then their demonstrations."""
    out = {}
    for binding in focus:
        failed = [row['id'] for row in view['items'] if row['kind'] == 'attempt' and not row['passed']
                  and row['failingBinding'] == binding]
        shown = [row['id'] for row in view['items'] if row['kind'] == 'demonstration' and row['binding'] == binding]
        out[binding] = (failed + shown)[:limit]
    return out


def focus_baseline(view, limit=3):
    """A crisp `focus` arm for the no-analysis ladder: bindings by failure count."""
    counts = view['facts']['failuresByBinding']
    return [binding for binding, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))][:limit]


def compact_rows(view, full_ids, bindings_under_revision):
    """Text rows for the optimizer context: full items for ``full_ids``; one line otherwise. Model-facing text
    (plans/MODEL_FACING_CHANGES.md records it until measured live)."""
    revise = set(bindings_under_revision)
    lines = []
    for row in view['items']:
        if row['kind'] == 'attempt':
            head = f"attempt {row['root']}: {'passed' if row['passed'] else 'failed'}"
            if not row['passed']:
                head += f" at {row['failingBinding']} ({row['errorKind']})"
            lines.append(head)
            if row['id'] in full_ids:
                for action in row['actions']:
                    if action['binding'] in revise:
                        for call, observation in zip(action['calls'], action['observations']):
                            lines.append(f"  {action['binding']} called {call[0]}({_short(call[1])}) -> {_short(observation)}")
        elif row['kind'] == 'demonstration':
            if row['id'] in full_ids:
                calls = '; '.join(f'{name}({_short(arguments)})' for name, arguments in row['calls'])
                lines.append(f"demonstration {row['root']} at {row['binding']}: {calls or _short(row['text'])}")
            else:
                lines.append(f"demonstration {row['root']} at {row['binding']}")
        else:
            gain = row['supportGain'] or {}
            lines.append(f"earlier step {row['step']} ({row['operator']}): support effect {gain.get('effect')}")
    return lines


def _short(value, limit=160):
    import json
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= limit else text[:limit - 3] + '...'
