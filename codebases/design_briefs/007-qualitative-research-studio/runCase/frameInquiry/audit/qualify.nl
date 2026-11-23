---
args: { draft: DraftInquiryFrame, candidate: ResearchProject, evidence: EvidencePage[], policy: PolicySelection }
returns: GateVerdict
readout: decision
---
1. Apply the application acceptance gate to this candidate (draft: draft, candidate: ResearchProject, evidence: EvidencePage[], policy: PolicySelection): Every subquestion has observable evidence and an explicit inferential limit.
2. Return "pass" only if source evidence (evidence) establishes each required condition for this candidate (candidate), with the applicable authority (policy), effective period, and scope.
3. Return "fail" if decisive evidence (evidence) establishes that a required condition is false or the draft (draft) contradicts an authoritative in-scope record (policy).
4. Return "needs-review" if required evidence (evidence) is missing, contradictory, out of date, outside scope, or lacks established authority (policy). Do not turn unsearched material into negative evidence.
