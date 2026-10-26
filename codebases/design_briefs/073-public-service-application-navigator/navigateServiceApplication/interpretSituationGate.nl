---
description: Exact counterfactual gate for attribution of applicant facts and uncertainty.
args: { evidence: EvidenceSlice; draft: DraftServiceSituation }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when stated needs, known facts, unknown facts and urgent routing are attributable to supplied records. Fail when an unsupported eligibility claim is introduced or an applicant fact is changed. Return needs-review when identity, authority, scope or current evidence is missing. Return only the finite verdict.
