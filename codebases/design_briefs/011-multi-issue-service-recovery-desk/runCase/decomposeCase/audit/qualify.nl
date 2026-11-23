---
args: { draft: DraftServiceIssue, candidate: RecoveryCase, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to this candidate (candidate: RecoveryCase): weigh whether every issue links to the relevant order and the customer’s stated outcome, using the draft (draft: DraftServiceIssue) and policy (policy: PolicySelection) to determine relevance.
2. Return "pass" only if source evidence (evidence: EvidencePage[]) establishes each required condition for the candidate (candidate: RecoveryCase), with the applicable authority (policy: PolicySelection), effective period, and scope.
3. Return "fail" if decisive evidence establishes that a required condition is false or the draft (draft: DraftServiceIssue) contradicts an authoritative in-scope record (policy: PolicySelection) for the candidate (candidate: RecoveryCase).
4. Return "needs-review" if required evidence is missing, contradictory, out of date, outside scope, or lacks established authority (policy: PolicySelection) for the candidate (candidate: RecoveryCase), using the draft (draft: DraftServiceIssue) and evidence (evidence: EvidencePage[]). Do not turn unsearched material into negative evidence.
