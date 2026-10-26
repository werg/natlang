---
description: Exact counterfactual gate for transition logistics requirements and their responsible roles.
args: { evidence: EvidenceSlice; draft: DraftTransitionRequirement }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when the logistical outcome, plan reference, prerequisite and responsible role are explicit and supported by the approved plan. Fail when a required clinical review or recipient acknowledgment is treated as completed without evidence. Return needs-review when role, prerequisite or current authority is missing. Return only the finite verdict.
