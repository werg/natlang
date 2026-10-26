---
description: Exact counterfactual gate for speaker attribution, transcript span and recollection uncertainty.
args: { evidence: EvidenceSlice; draft: DraftOralHistoryAccount }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when speaker, interview span, event description and remembered certainty language are preserved from the transcript. Fail when an uncertain recollection is asserted as verified fact or attributed to a different speaker. Return needs-review when the source span or consent status is missing. Return only the finite verdict.
