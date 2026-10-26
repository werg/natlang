---
description: Exact counterfactual gate for accepted terms and participant-level agreement status.
args: { evidence: EvidenceSlice; draft: DraftExistingAgreement }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when accepted terms, participants and effective window are supported by explicit current consent. Fail when the draft treats nonresponse as acceptance or overwrites a disputed prior term. Return needs-review when source authority or participant-specific acceptance is missing. Return only the finite verdict.
