---
description: Prepare an auditable Persistent World Game Director proposal.
args:
  request: string
  records: GameWorld[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['interpretIntent', 'modelCharacterViews', 'chooseWorldActions', 'advanceConsequences', 'maintainContinuity'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind interpretIntentOutput to interpretIntent({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind modelCharacterViewsOutput to modelCharacterViews({ goal: goal.outcome, records: interpretIntentOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind chooseWorldActionsOutput to chooseWorldActions({ goal: goal.outcome, records: modelCharacterViewsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind advanceConsequencesOutput to advanceConsequences({ goal: goal.outcome, records: chooseWorldActionsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind maintainContinuityOutput to maintainContinuity({ goal: goal.outcome, records: advanceConsequencesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { interpretIntent: interpretIntentOutput, modelCharacterViews: modelCharacterViewsOutput, chooseWorldActions: chooseWorldActionsOutput, advanceConsequences: advanceConsequencesOutput, maintainContinuity: maintainContinuityOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...interpretIntentOutput.nextQuestions, ...modelCharacterViewsOutput.nextQuestions, ...chooseWorldActionsOutput.nextQuestions, ...advanceConsequencesOutput.nextQuestions, ...maintainContinuityOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { interpretIntent: interpretIntentOutput, modelCharacterViews: modelCharacterViewsOutput, chooseWorldActions: chooseWorldActionsOutput, advanceConsequences: advanceConsequencesOutput, maintainContinuity: maintainContinuityOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a player disconnects mid-action, commit or reject once using the action identity. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show two characters spend the same resource, serialize the exact world transaction. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a character hears a rumor, store it as a belief rather than world truth. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a participant changes content boundaries, apply the restriction immediately to future generation. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a world rule changes between sessions, migrate at a checkpoint and preserve historical outcomes. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
