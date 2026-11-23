---
args: { draft: DraftCommitment, candidate: Handover, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate (Handover): every commitment has an attributable statement; missing dates stay unknown rather than invented.
2. Return "pass" only if source evidence (EvidencePage[]) establishes each required condition for the candidate (Handover), with the applicable authority, effective period, and scope.
3. Return "fail" if decisive evidence establishes that a required condition is false or the draft (DraftCommitment) contradicts an authoritative in-scope record.
4. Return "needs-review" if required evidence is missing, contradictory, out of date, outside scope, or lacks established authority. Do not turn unsearched material into negative evidence.
