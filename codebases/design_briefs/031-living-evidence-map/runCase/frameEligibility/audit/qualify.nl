---
args: { draft: DraftEligibilityCriterion, candidate: EvidenceReview, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate: weigh whether the criteria are frozen by version and usable on unseen records, using the draft's eligibility criteria and the candidate's evidence, and apply the policy selection.
2. Return "pass" only if source evidence establishes each required condition for the candidate, with the applicable authority, effective period, and scope; verify the evidence against the policy and draft.
3. Return "fail" if decisive evidence establishes that a required condition is false or the draft contradicts an authoritative in-scope record; compare the draft to authoritative records within the policy.
4. Return "needs-review" if required evidence is missing, contradictory, out of date, outside scope, or lacks established authority; do not treat unsearched material as negative evidence; assess the evidence against the policy and draft.
