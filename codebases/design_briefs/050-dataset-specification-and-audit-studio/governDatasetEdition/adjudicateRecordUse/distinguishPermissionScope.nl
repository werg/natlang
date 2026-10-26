---
args: {"record": "DatasetCandidate", "use": "DatasetUse", "events": "Evidence[]"}
returns: "RecordEligibility"
---
Trace each claimed permission to its issuer, subject, purpose, audience, artifact form and effective period. An absent consent or license is unknown, not permissive. Determine whether research use, model training, derivative release and source redistribution are separately authorized. Preserve a specific prohibition even when another use is allowed. Bind the eligibility finding to exact evidence identities and leave family/slice identities for the caller's established mappings.
