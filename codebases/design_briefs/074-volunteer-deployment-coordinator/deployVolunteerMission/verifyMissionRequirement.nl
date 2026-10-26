---
description: Exact counterfactual gate for mission capability and completion requirements.
args: { evidence: EvidenceSlice; draft: DraftMissionRequirement }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when the task, required skill, access condition and completion evidence are explicit and attributable to current mission authority. Fail when the proposed team plan weakens a required qualification to meet headcount. Return needs-review when a qualification, scope or current source is missing. Return only the finite verdict.
