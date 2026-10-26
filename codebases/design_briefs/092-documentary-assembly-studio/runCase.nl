---
description: Prepare an auditable Documentary Assembly Studio proposal.
args:
  request: string
  records: DocumentaryProject[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['indexMaterial', 'constructArguments', 'selectEdits', 'reviewProduction', 'freezeAssembly'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind indexMaterialOutput to indexMaterial({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind constructArgumentsOutput to constructArguments({ goal: goal.outcome, records: indexMaterialOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind selectEditsOutput to selectEdits({ goal: goal.outcome, records: constructArgumentsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind reviewProductionOutput to reviewProduction({ goal: goal.outcome, records: selectEditsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind freezeAssemblyOutput to freezeAssembly({ goal: goal.outcome, records: reviewProductionOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { indexMaterial: indexMaterialOutput, constructArguments: constructArgumentsOutput, selectEdits: selectEditsOutput, reviewProduction: reviewProductionOutput, freezeAssembly: freezeAssemblyOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...indexMaterialOutput.nextQuestions, ...constructArgumentsOutput.nextQuestions, ...selectEditsOutput.nextQuestions, ...reviewProductionOutput.nextQuestions, ...freezeAssemblyOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { indexMaterial: indexMaterialOutput, constructArguments: constructArgumentsOutput, selectEdits: selectEditsOutput, reviewProduction: reviewProductionOutput, freezeAssembly: freezeAssemblyOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a short quote loses its qualifier, restore context or reject the cut. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show two adjacent clips imply causation, require evidence or revise the transition. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show archive rights expire before release, obtain renewal or replace the asset. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a speaker corrects a factual statement, preserve the correction in editorial review. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a late edit shifts duration, update captions, descriptions, and clearance references together. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
