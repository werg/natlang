---
description: Exact counterfactual gate for preparation instruction source, version and timing rule.
args: { evidence: EvidenceSlice; draft: DraftPreparationRequirement }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when the approved instruction reference, required action, timing rule and exception are supported by current authority. Fail when a reminder changes an exception or reverses a clinical instruction. Return needs-review when version, timezone, authority or source is missing. Return only the finite verdict.
