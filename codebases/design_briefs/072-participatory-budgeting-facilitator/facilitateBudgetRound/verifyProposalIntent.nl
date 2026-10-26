---
description: Exact counterfactual gate for proposal intent and beneficiary purpose.
args: { evidence: EvidenceSlice; draft: DraftCivicProposal }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when the proposal's intended benefit, required work and beneficiary scope are attributable to its proposer and current evidence. Fail when normalization silently changes the project's purpose. Return needs-review when attribution, scope, or decisive evidence is missing or conflicting. Return only the finite verdict.
