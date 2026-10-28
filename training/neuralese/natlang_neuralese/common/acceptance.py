"""One acceptance gate: select on development, confirm once on untouched confirmation worlds.

Shared by stored guidance (memetic learner), program tunes and bundles, learned-optimizer proposals and
checkpoint selection. Decomposition: ``DECOMPOSITION.md`` (this directory), part 3.

A measurement is the actual scored task outcome of one artifact on one slice: ``{'successes': int,
'scored': int}`` (extra fields are ignored), measured with paired seeds and limits for every artifact
compared on that slice.

1. ``select(parent, candidate)``: development only. A candidate replaces the selected parent when it
   solves strictly more of the same number of scored rollouts. Confirmation outcomes never enter selection.
2. ``confirm(...)``: the selected artifact against the baseline on development and on the confirmation
   slice, each with at least ``min_gain_fraction`` absolute success-rate gain. A ledger spends the
   confirmation slice on its first confirmed artifact; confirming a different artifact on a spent slice is
   refused (those worlds are no longer untouched). The receipt reports the actual outcomes of both arms.
"""
from __future__ import annotations

import json
from pathlib import Path

from .hashing import canonical_json_sha256_hex
from .jsonio import write_json_durable

DECISION_SCHEMA = 'natlang.acceptance-decision/1'
LEDGER_SCHEMA = 'natlang.confirmation-ledger/1'
SLICES = ('development', 'confirmation')


class AcceptanceError(ValueError):
    """An adoption would use confirmation worlds for selection, reuse spent ones, or compare unpaired outcomes."""


def outcome(measurement):
    successes, scored = measurement.get('successes'), measurement.get('scored')
    if type(successes) is not int or type(scored) is not int or not 0 <= successes <= scored:
        raise AcceptanceError('a measurement is the actual scored task outcome {successes, scored}')
    return {'successes': successes, 'scored': scored}


def select(parent, candidate):
    """Development selection on paired outcomes: same scored count, strictly more successes."""
    parent, candidate = outcome(parent), outcome(candidate)
    return candidate['scored'] == parent['scored'] > 0 and candidate['successes'] > parent['successes']


def gain(baseline, selected):
    baseline, selected = outcome(baseline), outcome(selected)
    if selected['scored'] != baseline['scored'] or not selected['scored']:
        return None
    return (selected['successes'] - baseline['successes']) / selected['scored']


def _ledger(path):
    path = Path(path)
    value = json.loads(path.read_text()) if path.exists() else {'schema': LEDGER_SCHEMA, 'spent': {}}
    if value.get('schema') != LEDGER_SCHEMA or not isinstance(value.get('spent'), dict):
        raise AcceptanceError('confirmation ledger is invalid')
    return value


def require_untouched(confirmation, artifact, ledger):
    """Before measuring: refuse confirmation worlds already spent on a different artifact."""
    spent = _ledger(ledger)['spent'].get(canonical_json_sha256_hex(confirmation))
    if spent is not None and spent['artifact'] != artifact:
        raise AcceptanceError('confirmation worlds were already spent on a different artifact; '
                              'confirm on fresh untouched worlds')


def confirm(*, artifact, baseline, selected, min_gain_fraction, confirmation, ledger):
    """Decide adoption of the development-selected ``artifact`` and spend its confirmation slice.

    ``baseline``/``selected``: {'development': measurement, 'confirmation': measurement}.
    ``artifact``: exact identity of the selected artifact (state id, release digest, checkpoint sha...).
    ``confirmation``: exact binding of the confirmation slice (e.g. the custody slice receipt entry).
    """
    if not isinstance(artifact, str) or not artifact:
        raise AcceptanceError('confirmation needs the exact selected artifact identity')
    if not 0 < min_gain_fraction <= 1:
        raise AcceptanceError('min_gain_fraction must be in (0, 1]')
    slice_sha256 = canonical_json_sha256_hex(confirmation)
    gains = {name: gain(baseline[name], selected[name]) for name in SLICES}
    qualified = all(value is not None and value >= min_gain_fraction for value in gains.values())
    receipt = {'schema': DECISION_SCHEMA, 'artifact': artifact, 'confirmation_slice_sha256': slice_sha256,
               'status': 'qualified' if qualified else 'no_lift', 'min_gain_fraction': min_gain_fraction,
               'baseline': {name: outcome(baseline[name]) for name in SLICES},
               'selected': {name: outcome(selected[name]) for name in SLICES}, 'gain': gains}
    book = _ledger(ledger)
    spent = book['spent'].get(slice_sha256)
    if spent is not None:
        if spent['artifact'] != artifact:
            raise AcceptanceError('confirmation worlds were already spent on a different artifact; '
                                  'confirm on fresh untouched worlds')
        if spent['decision'] != receipt:
            raise AcceptanceError('re-confirmation of the same artifact changed its measured outcome')
        return receipt
    book['spent'][slice_sha256] = {'artifact': artifact, 'decision': receipt}
    write_json_durable(Path(ledger), book)
    return receipt
