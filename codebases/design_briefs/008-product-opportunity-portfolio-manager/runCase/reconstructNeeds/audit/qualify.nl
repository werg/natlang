---
args: { draft: DraftCustomerNeed, candidate: OpportunityPortfolio, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate (candidate: OpportunityPortfolio): every need must have attributable evidence; do not invent segments or use unsupported market sizes.
2. Return "pass" only if source evidence (evidence: EvidencePage[]) establishes each required condition for the candidate (candidate: OpportunityPortfolio), citing the applicable authority, effective period, and scope.
3. Return "fail" if decisive evidence shows a required condition is false or the draft (draft: DraftCustomerNeed) contradicts an authoritative in-scope record.
4. Return "needs-review" if required evidence (evidence: EvidencePage[]) is missing, contradictory, out of date, outside scope, or lacks established authority (policy: PolicySelection); do not treat unsearched material as negative evidence.
