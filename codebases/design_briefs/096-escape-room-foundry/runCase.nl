---
description: Prepare an auditable Escape Room Foundry proposal.
args:
  request: string
  records: EscapeRoomProject[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['modelExperience', 'constructPuzzleGraph', 'writeArtifacts', 'simulatePlayers', 'packageOperations'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind modelExperienceOutput to modelExperience({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind constructPuzzleGraphOutput to constructPuzzleGraph({ goal: goal.outcome, records: modelExperienceOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind writeArtifactsOutput to writeArtifacts({ goal: goal.outcome, records: constructPuzzleGraphOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind simulatePlayersOutput to simulatePlayers({ goal: goal.outcome, records: writeArtifactsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind packageOperationsOutput to packageOperations({ goal: goal.outcome, records: simulatePlayersOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { modelExperience: modelExperienceOutput, constructPuzzleGraph: constructPuzzleGraphOutput, writeArtifacts: writeArtifactsOutput, simulatePlayers: simulatePlayersOutput, packageOperations: packageOperationsOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...modelExperienceOutput.nextQuestions, ...constructPuzzleGraphOutput.nextQuestions, ...writeArtifactsOutput.nextQuestions, ...simulatePlayersOutput.nextQuestions, ...packageOperationsOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { modelExperience: modelExperienceOutput, constructPuzzleGraph: constructPuzzleGraphOutput, writeArtifacts: writeArtifactsOutput, simulatePlayers: simulatePlayersOutput, packageOperations: packageOperationsOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a team solves a puzzle before finding its clue, ensure the world state remains coherent. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a prop is lost or broken, provide an approved staff recovery path without trapping participants. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a reset misses one item, detect it through a pre-session checklist and state verification. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a participant declines a sensory element, offer an equivalent accessible route. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a clue has two reasonable answers, repair the artifact or accept both when consistent with the design. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
