---
description: Prepare an auditable Plain Language Transformation Studio proposal.
args:
  request: string
  records: PlainLanguageProject[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['recoverMeaning', 'modelReaderTasks', 'rewriteStructure', 'challengeComprehension', 'releaseAccessible'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind recoverMeaningOutput to recoverMeaning({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind modelReaderTasksOutput to modelReaderTasks({ goal: goal.outcome, records: recoverMeaningOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind rewriteStructureOutput to rewriteStructure({ goal: goal.outcome, records: modelReaderTasksOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind challengeComprehensionOutput to challengeComprehension({ goal: goal.outcome, records: rewriteStructureOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind releaseAccessibleOutput to releaseAccessible({ goal: goal.outcome, records: challengeComprehensionOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { recoverMeaning: recoverMeaningOutput, modelReaderTasks: modelReaderTasksOutput, rewriteStructure: rewriteStructureOutput, challengeComprehension: challengeComprehensionOutput, releaseAccessible: releaseAccessibleOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...recoverMeaningOutput.nextQuestions, ...modelReaderTasksOutput.nextQuestions, ...rewriteStructureOutput.nextQuestions, ...challengeComprehensionOutput.nextQuestions, ...releaseAccessibleOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { recoverMeaning: recoverMeaningOutput, modelReaderTasks: modelReaderTasksOutput, rewriteStructure: rewriteStructureOutput, challengeComprehension: challengeComprehensionOutput, releaseAccessible: releaseAccessibleOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a summary removes a crucial unless clause, block release through a boundary-case test. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show an example sounds like a new rule, label and revise it to preserve scope. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a technical term has no safe plain synonym, retain it with an explanation. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show the source changes after review, invalidate only affected meaning units and approvals. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show readers use assistive technology, verify headings, reading order, and link descriptions. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
