"""Verifier for applications/semantic-etl bench units (Tier 1 flagship A2)."""

import math
import re

ISO_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
ISO_CCY = re.compile(r'^[A-Z]{3}$')


def verify(expected, output):
    """Return `{passed, checks, safety}` for one Semantic ETL batch unit."""
    if not isinstance(output, dict) or not isinstance(output.get('rows'), list) or not isinstance(output.get('conservation'), dict):
        return {
            'passed': False,
            'checks': {'row_count': False, 'ids_preserved': False, 'schema_valid': False, 'total_conserved': False},
            'safety': {'schema_violation': 1, 'conservation_drift': 1},
        }

    rows = output['rows']
    want_ids = expected['inputIds']
    row_count_ok = len(rows) == len(want_ids)
    got_ids = [r.get('id') for r in rows if isinstance(r, dict)]
    ids_ok = set(got_ids) == set(want_ids) and len(got_ids) == len(want_ids)

    schema_violations = 0
    total = 0.0
    for row in rows:
        if not isinstance(row, dict):
            schema_violations += 1
            continue
        amt = row.get('amount')
        ccy = row.get('currency')
        date = row.get('date')
        vendor = row.get('vendor')
        cat = row.get('category')
        if (
            not isinstance(amt, (int, float)) or isinstance(amt, bool) or not math.isfinite(amt)
            or not isinstance(ccy, str) or not ISO_CCY.match(ccy)
            or not isinstance(date, str) or not ISO_DATE.match(date)
            or not isinstance(vendor, str) or not vendor.strip()
            or not isinstance(cat, str) or not cat.strip()
        ):
            schema_violations += 1
        else:
            total += float(amt)

    schema_ok = schema_violations == 0
    total_ok = abs(round(total, 2) - float(expected['expectedTotal'])) <= 0.01
    conservation_ok = bool(output['conservation'].get('ok')) and total_ok

    passed = row_count_ok and ids_ok and schema_ok and conservation_ok
    return {
        'passed': passed,
        'checks': {
            'row_count': row_count_ok,
            'ids_preserved': ids_ok,
            'schema_valid': schema_ok,
            'total_conserved': total_ok,
        },
        'safety': {
            'schema_violation': schema_violations,
            'conservation_drift': 0 if total_ok else 1,
        },
    }
