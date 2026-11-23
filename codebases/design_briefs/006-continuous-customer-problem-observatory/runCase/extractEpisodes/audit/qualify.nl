---
args: { draft: DraftProblemEpisode, candidate: ProblemStudy, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate (candidate: ProblemStudy). Every episode preserves the customer's actual situation and does not invent severity; weigh the candidate's actual situation against any invented severity.

2. Return "pass" only if source evidence establishes each required condition for the candidate (candidate: ProblemStudy), including the applicable authority, effective period, and scope (evidence: EvidencePage[], policy: PolicySelection).

3. Return "fail" if decisive evidence establishes that a required condition is false or the draft contradicts an authoritative in-scope record (draft: DraftProblemEpisode).

4. Return "needs-review" if required evidence is missing, contradictory, out of date, outside scope, or lacks established authority (policy: PolicySelection). Do not treat unsearched material as negative evidence.
