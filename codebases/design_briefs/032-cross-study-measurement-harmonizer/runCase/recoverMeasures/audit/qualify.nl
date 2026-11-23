---
args: { draft: DraftMeasureDefinition, candidate: Harmonization, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate (candidate: Harmonization): No mapping based only on column name or numeric range. Consider the draft (draft: DraftMeasureDefinition), the policy (policy: PolicySelection), and the evidence (evidence: EvidencePage[]) when applying the gate.
2. Return "pass" only if source evidence (evidence: EvidencePage[]) establishes each required condition for the candidate (candidate: Harmonization), with the applicable authority, effective period, and scope. Take into account the draft and policy.
3. Return "fail" if decisive evidence establishes that a required condition is false or the draft (draft: DraftMeasureDefinition) contradicts an authoritative in-scope record for the candidate (candidate: Harmonization).
4. Return "needs-review" if required evidence (evidence: EvidencePage[]) is missing, contradictory, out of date, outside scope, or lacks established authority. Do not treat unsearched material as negative evidence. Consider the candidate (candidate: Harmonization), the draft (draft: DraftMeasureDefinition), and the policy (policy: PolicySelection).
