---
args: { draft: DraftAmendmentTarget, candidate: AmendmentCase, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate. Weigh the draft's target identity and base edition against the candidate's prior state; if the draft's target identity and base edition were established before transformation, the gate applies. Use the draft and candidate arguments from the signature.

2. Return "pass" only if source evidence establishes each required condition for this candidate, with the applicable authority, effective period, and scope. Examine the evidence pages for the candidate and confirm that the policy selection supports the conditions. If all conditions are supported, return pass.

3. Return "fail" if decisive evidence establishes that a required condition is false or the draft contradicts an authoritative in-scope record. Compare the draft against authoritative records within the candidate's scope; if a contradiction exists, return fail.

4. Return "needs-review" if required evidence is missing, contradictory, out of date, outside scope, or lacks established authority. Do not treat unsearched material as negative evidence. If any of these apply, return needs-review.
