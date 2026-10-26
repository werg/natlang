---
args: { request: SubstituteEvaluationRequest, changes: EvidencePage[], candidates: StageResultSubstituteCandidate }
returns: QualificationGuardrail
---
Compare supplier identity, part number and revision in each change with each candidate's source revision before compatibility analysis. Exclude superseded candidate evidence from compatibility proof while retaining its history; carry only unaffected or freshly sourced revisions forward.
