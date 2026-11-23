---
args: { draft: DraftScopedClaim, candidate: KnowledgeRevision, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to the candidate (KnowledgeRevision): verify that the draft (DraftScopedClaim) retains every material negation, exception, and audience qualifier. 2. Return "pass" only if the source evidence (EvidencePage[]) establishes each required condition for the candidate (KnowledgeRevision), with the applicable authority, effective period, and scope. 3. Return "fail" if decisive evidence establishes that a required condition is false or the draft (DraftScopedClaim) contradicts an authoritative in-scope record. 4. Return "needs-review" if required evidence is missing, contradictory, out of date, outside scope, or lacks established authority (policy: PolicySelection); do not turn unsearched material into negative evidence.
