---
description: Prepare an auditable Magazine Commissioning And Issue Planning Desk proposal.
args:
  request: string
  records: MagazineIssue[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['interpretPitches', 'mapCoverage', 'commissionPieces', 'manageEditorialWork', 'qualifyIssue'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind interpretPitchesOutput to interpretPitches({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind mapCoverageOutput to mapCoverage({ goal: goal.outcome, records: interpretPitchesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind commissionPiecesOutput to commissionPieces({ goal: goal.outcome, records: mapCoverageOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind manageEditorialWorkOutput to manageEditorialWork({ goal: goal.outcome, records: commissionPiecesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind qualifyIssueOutput to qualifyIssue({ goal: goal.outcome, records: manageEditorialWorkOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { interpretPitches: interpretPitchesOutput, mapCoverage: mapCoverageOutput, commissionPieces: commissionPiecesOutput, manageEditorialWork: manageEditorialWorkOutput, qualifyIssue: qualifyIssueOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...interpretPitchesOutput.nextQuestions, ...mapCoverageOutput.nextQuestions, ...commissionPiecesOutput.nextQuestions, ...manageEditorialWorkOutput.nextQuestions, ...qualifyIssueOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { interpretPitches: interpretPitchesOutput, mapCoverage: mapCoverageOutput, commissionPieces: commissionPiecesOutput, manageEditorialWork: manageEditorialWorkOutput, qualifyIssue: qualifyIssueOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a contributor loses access to a key source, revise the commission rather than pretending the reporting plan still holds. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show two pieces cover the same angle, negotiate differentiation before drafting duplicates. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a headline overstates the article, include it in factual review. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a late withdrawal leaves a section gap, use a prequalified contingency piece. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a photo credit changes, update print, web, and promotional assets consistently. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
