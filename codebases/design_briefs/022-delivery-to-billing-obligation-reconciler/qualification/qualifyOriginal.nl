---
args: { draft: DraftBillingCondition, candidate: BillingCase, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to this candidate (candidate): every billable condition has a source and an effective scope. Weigh the evidence (evidence) against the draft (draft) and policy (policy); the draft and policy are the authoritative sources for scope and authority.
2. Return "pass" only if the source evidence (evidence) establishes each required condition for this candidate, citing the applicable authority (policy), the effective period, and the scope.
3. Return "fail" if decisive evidence (evidence) establishes that a required condition is false or the draft (draft) contradicts an authoritative in-scope record (policy).
4. Return "needs-review" if required evidence (evidence) is missing, contradictory, out of date, outside scope, or lacks established authority (policy); do not treat unsearched material as negative evidence.
