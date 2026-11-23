---
args: { draft: DraftTenderRequirement, candidate: TenderResponse, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate (candidate: TenderResponse): map each source requirement to a response slot or an explicit nonresponse decision, weighing the draft (draft: DraftTenderRequirement), the evidence (evidence: EvidencePage[]) and the policy (policy: PolicySelection) as relevant.
2. Return "pass" only if the evidence (evidence: EvidencePage[]) establishes each required condition for the candidate (candidate: TenderResponse), citing the applicable authority, effective period, and scope (policy: PolicySelection).
3. Return "fail" if decisive evidence (evidence: EvidencePage[]) establishes that a required condition is false, or if the draft (draft: DraftTenderRequirement) contradicts an authoritative in-scope record (policy: PolicySelection).
4. Return "needs-review" if required evidence (evidence: EvidencePage[]) is missing, contradictory, out of date, outside scope, or lacks established authority (policy: PolicySelection). Do not treat unsearched material as negative evidence.
