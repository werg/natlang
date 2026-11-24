"""Verifier for applications/sem-sheet bench units (Tier 1 flagship A3)."""

VALID_LABELS = ('high', 'medium', 'low', 'none')


def verify(expected, output):
    """Return `{passed, checks, safety}` for one Sem-Sheet evaluation unit."""
    if not isinstance(output, dict) or not isinstance(output.get('values'), dict):
        return {
            'passed': False,
            'checks': {'values': False, 'labels': False, 'layers': False},
            'safety': {'schema_violation': 1},
        }

    values = output['values']
    values_ok = True
    for coord, want in (expected.get('expectedValues') or {}).items():
        got = values.get(coord)
        if isinstance(want, (int, float)) and not isinstance(want, bool):
            if not isinstance(got, (int, float)) or isinstance(got, bool) or abs(float(got) - float(want)) > 1e-3:
                values_ok = False
        elif got != want:
            values_ok = False

    labels_ok = True
    schema_violations = 0
    for coord, want in (expected.get('expectedLabels') or {}).items():
        got = values.get(coord)
        if got not in VALID_LABELS:
            schema_violations += 1
        if got != want:
            labels_ok = False

    non_empty_ok = True
    for coord in expected.get('nonEmptyCells') or []:
        got = values.get(coord)
        if not isinstance(got, str) or not got.strip():
            non_empty_ok = False

    layers_ok = True
    if 'expectedLayers' in expected and isinstance(output.get('layers'), list):
        layers_ok = len(output['layers']) == expected['expectedLayers']

    passed = values_ok and labels_ok and non_empty_ok and layers_ok and schema_violations == 0
    return {
        'passed': passed,
        'checks': {
            'values': values_ok,
            'labels': labels_ok,
            'non_empty': non_empty_ok,
            'layers': layers_ok,
        },
        'safety': {'schema_violation': schema_violations},
    }
