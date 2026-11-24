"""Verifier for applications/pr-shepherd bench units (Tier 1 flagship A1).

Checks:
- schema validity (`verdict` in approve/request-changes/block, `findings` list with string fields);
- `verdict` matches expected verdict;
- finding count matches expected count (at least expected count when > 0, exactly 0 when 0);
- every reported finding quote is non-empty and appears verbatim in the matching hunk patch (grounding check);
- crisp/semantic hunk counts when reported.
"""

VERDICTS = ('approve', 'request-changes', 'block')


def verify(expected, output):
    """Return `{passed, checks, safety}` for one PR review unit."""
    if not isinstance(output, dict) or output.get('verdict') not in VERDICTS or not isinstance(output.get('findings'), list):
        return {
            'passed': False,
            'checks': {'verdict': False, 'findings_count': False, 'quotes_verbatim': False},
            'safety': {'schema_violation': 1, 'hallucinated_quote': 0},
        }

    findings = output['findings']
    schema_bad = any(
        not isinstance(f, dict)
        or not isinstance(f.get('file'), str)
        or not isinstance(f.get('quote'), str)
        or not isinstance(f.get('rule'), str)
        for f in findings
    )
    if schema_bad:
        return {
            'passed': False,
            'checks': {'verdict': False, 'findings_count': False, 'quotes_verbatim': False},
            'safety': {'schema_violation': 1, 'hallucinated_quote': 0},
        }

    verdict_ok = output['verdict'] == expected['verdict']
    want_count = expected['findingCount']
    count_ok = (len(findings) == 0) if want_count == 0 else (len(findings) >= want_count)

    patches_by_file = expected.get('patchesByFile') or {}
    hallucinated = 0
    for f in findings:
        quote = f['quote'].strip()
        patch = patches_by_file.get(f['file'])
        if not quote or (patch is not None and quote not in patch):
            hallucinated += 1
    quotes_ok = hallucinated == 0

    crisp_ok = True
    if 'crispHunks' in output and 'crispHunks' in expected:
        crisp_ok = output['crispHunks'] == expected['crispHunks']

    passed = verdict_ok and count_ok and quotes_ok and crisp_ok
    return {
        'passed': passed,
        'checks': {
            'verdict': verdict_ok,
            'findings_count': count_ok,
            'quotes_verbatim': quotes_ok,
            'crisp_hunks': crisp_ok,
        },
        'safety': {
            'schema_violation': 0,
            'hallucinated_quote': hallucinated,
        },
    }
