"""The optimizer's request rendering (model-facing; recorded in plans/MODEL_FACING_CHANGES.md until measured live).

The program contract and the current bundle are rendered once; evidence appears as compact rows with full items only
for the selected key evidence. Current instruction/skill states and evidence digests enter as native parts, so the
writer consumes them natively. The long input carries zero prompt weight: the optimizer's token losses cover its
outputs only.
"""
from __future__ import annotations

REQUEST_SCHEMA = 'natlang.lopt.request/1'


def request_messages(*, contract, bundle_lines, evidence_lines, current_states, digests=(), focus=()):
    """``current_states``: [(label, state id)] of bindings under revision; ``digests``: [(binding, state id)]."""
    content = [{'type': 'text', 'text': f'Program:\n{contract}\n\nCurrent adaptation:\n' + '\n'.join(bundle_lines) + '\n'}]
    for label, state in current_states:
        content.append({'type': 'text', 'text': f'{label}: '})
        content.append({'type': 'native_state', 'id': state})
        content.append({'type': 'text', 'text': '\n'})
    if focus:
        content.append({'type': 'text', 'text': 'Revise: ' + ', '.join(focus) + '\n'})
    for binding, state in digests:
        content.append({'type': 'text', 'text': f'Missing behavior at {binding}: '})
        content.append({'type': 'native_state', 'id': state})
        content.append({'type': 'text', 'text': '\n'})
    content.append({'type': 'text', 'text': 'Evidence:\n' + '\n'.join(evidence_lines)})
    return [{'role': 'user', 'content': content}]
