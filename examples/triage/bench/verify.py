"""The triage bench verifier: a unit is one ticket, its output the program's Report for that one ticket.

A unit passes when the report gives the ticket its gold label (``by_label`` is exactly ``{label: 1}``, zero counts
ignored) and its gold urgency (``urgent`` is 1 or 0). The summary is a judgment with no oracle: it is recorded (word
count, whether it keeps the 60-word limit) but does not decide the pass. A report that is not a Report is a schema
violation.
"""

LABELS = ('billing', 'technical', 'spam')


def verify(expected, output):
    """``{passed, checks, safety}`` for one unit."""
    if not (isinstance(output, dict) and isinstance(output.get('urgent'), int) and not isinstance(output.get('urgent'), bool)
            and isinstance(output.get('by_label'), dict) and isinstance(output.get('summary'), str)
            and all(isinstance(v, int) and not isinstance(v, bool) for v in output['by_label'].values())):
        return {'passed': False, 'checks': {'label': False, 'urgent': False}, 'safety': {'schema_violation': 1}}
    labels = {k: v for k, v in output['by_label'].items() if v}
    label_ok = labels == {expected['label']: 1}
    urgent_ok = output['urgent'] == int(bool(expected['urgent']))
    words = len(output['summary'].split())
    return {'passed': label_ok and urgent_ok,
            'checks': {'label': label_ok, 'urgent': urgent_ok, 'summary_words': words, 'summary_within_60': words <= 60},
            'safety': {'schema_violation': int(any(k not in LABELS for k in labels))}}
