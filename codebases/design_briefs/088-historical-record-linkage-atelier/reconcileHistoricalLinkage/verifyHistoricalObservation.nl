---
description: Exact counterfactual gate for preserving historical record wording and date/place uncertainty.
args: { evidence: EvidenceSlice; draft: DraftHistoricalObservation }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when recorded name, date expression, place expression and source identity are preserved. Fail when an approximate date is converted to unsupported precision or historical geography is forced to a modern identity. Return needs-review when calendar, boundary or source authority is unresolved. Return only the finite verdict.
