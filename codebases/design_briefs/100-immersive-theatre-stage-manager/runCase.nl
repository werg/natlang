---
description: Prepare an auditable Immersive Theatre Stage Manager proposal.
args:
  request: string
  records: ImmersiveProduction[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['modelDramaturgy', 'modelProduction', 'selectSceneVariants', 'coordinatePerformance', 'recoverContinuity'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind modelDramaturgyOutput to modelDramaturgy({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind modelProductionOutput to modelProduction({ goal: goal.outcome, records: modelDramaturgyOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind selectSceneVariantsOutput to selectSceneVariants({ goal: goal.outcome, records: modelProductionOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind coordinatePerformanceOutput to coordinatePerformance({ goal: goal.outcome, records: selectSceneVariantsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind recoverContinuityOutput to recoverContinuity({ goal: goal.outcome, records: coordinatePerformanceOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { modelDramaturgy: modelDramaturgyOutput, modelProduction: modelProductionOutput, selectSceneVariants: selectSceneVariantsOutput, coordinatePerformance: coordinatePerformanceOutput, recoverContinuity: recoverContinuityOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...modelDramaturgyOutput.nextQuestions, ...modelProductionOutput.nextQuestions, ...selectSceneVariantsOutput.nextQuestions, ...coordinatePerformanceOutput.nextQuestions, ...recoverContinuityOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { modelDramaturgy: modelDramaturgyOutput, modelProduction: modelProductionOutput, selectSceneVariants: selectSceneVariantsOutput, coordinatePerformance: coordinatePerformanceOutput, recoverContinuity: recoverContinuityOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show an audience member declines participation, switch to an equivalent nonparticipatory path without pressure. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a cast member becomes unavailable, select a prequalified variant or obtain approval for a bounded rewrite. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a prop fails mid-scene, use an approved narrative recovery, never an improvised hazardous mechanism. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show two audience groups learn different facts, maintain separate knowledge state for later scenes. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a stop signal is given, halt relevant activity immediately under the production’s safety protocol. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
