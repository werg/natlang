---
args: { draft: DraftCloseTask, candidate: ReconciliationBreak, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
---
1. Apply the application acceptance gate to this candidate: Every material break has an accepted owner and completion criterion.
2. Return "pass" only if source evidence establishes each required condition for this candidate, with the applicable authority, effective period, and scope.
3. Return "fail" if decisive evidence establishes that a required condition is false or the draft contradicts an authoritative in-scope record.
4. Return "needs-review" if required evidence is missing, contradictory, out of date, outside scope, or lacks established authority. Do not turn unsearched material into negative evidence.
