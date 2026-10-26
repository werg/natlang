---
description: Exact counterfactual gate for recipient checklist requirements and authorized accepting roles.
args: { evidence: EvidenceSlice; draft: DraftReferralRequirement }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when required document, validity rule, service and accepting role are explicit in current recipient authority. Fail when a present but unsigned or wrong-episode document is treated as complete. Return needs-review when checklist version or authority is missing. Return only the finite verdict.
