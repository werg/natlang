---
args: { draft: DraftMethodStep, candidate: ReplicationProject, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate (ReplicationProject): weigh each decisive methodological choice against its source or explicit unknown status, using the draft (DraftMethodStep), candidate evidence (EvidencePage[]), and policy (PolicySelection) as relevant.

2. Return "pass" only if source evidence (EvidencePage[]) establishes each required condition for the candidate (ReplicationProject), with the applicable authority, effective period, and scope, considering the draft (DraftMethodStep) and policy (PolicySelection).

3. Return "fail" if decisive evidence (EvidencePage[]) establishes that a required condition is false or the draft (DraftMethodStep) contradicts an authoritative in-scope record for the candidate (ReplicationProject).

4. Return "needs-review" if required evidence (EvidencePage[]) is missing, contradictory, out of date, outside scope, or lacks established authority (PolicySelection). Do not treat unsearched material as negative evidence.
