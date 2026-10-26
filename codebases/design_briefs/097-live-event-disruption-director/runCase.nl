---
description: Prepare an auditable Live Event Disruption Director proposal.
args:
  request: string
  records: LiveEvent[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['recoverEventPromises', 'interpretFailure', 'generateFormats', 'rebuildRunOfShow', 'coordinateLiveChange'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind recoverEventPromisesOutput to recoverEventPromises({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind interpretFailureOutput to interpretFailure({ goal: goal.outcome, records: recoverEventPromisesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind generateFormatsOutput to generateFormats({ goal: goal.outcome, records: interpretFailureOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind rebuildRunOfShowOutput to rebuildRunOfShow({ goal: goal.outcome, records: generateFormatsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind coordinateLiveChangeOutput to coordinateLiveChange({ goal: goal.outcome, records: rebuildRunOfShowOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { recoverEventPromises: recoverEventPromisesOutput, interpretFailure: interpretFailureOutput, generateFormats: generateFormatsOutput, rebuildRunOfShow: rebuildRunOfShowOutput, coordinateLiveChange: coordinateLiveChangeOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...recoverEventPromisesOutput.nextQuestions, ...interpretFailureOutput.nextQuestions, ...generateFormatsOutput.nextQuestions, ...rebuildRunOfShowOutput.nextQuestions, ...coordinateLiveChangeOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { recoverEventPromises: recoverEventPromisesOutput, interpretFailure: interpretFailureOutput, generateFormats: generateFormatsOutput, rebuildRunOfShow: rebuildRunOfShowOutput, coordinateLiveChange: coordinateLiveChangeOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a speaker is delayed but not absent, preserve an explicit decision deadline for switching formats. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a technical fallback lacks accessibility support, reject it or add an approved equivalent. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show staff receive conflicting schedule versions, show revision IDs and require acknowledgment. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a sponsor commitment changes, route commercial approval separately from stage coordination. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show an emergency occurs, suspend generated replanning and defer to the approved emergency command process. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
