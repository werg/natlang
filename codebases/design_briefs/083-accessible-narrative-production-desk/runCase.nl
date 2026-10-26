---
description: Prepare an auditable Accessible Narrative Production Desk proposal.
args:
  request: string
  records: AccessibleProduction[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['modelNarrativeInformation', 'planChannels', 'writeAccessibleText', 'validateSynchronization', 'packageAccessibleMedia'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind modelNarrativeInformationOutput to modelNarrativeInformation({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind planChannelsOutput to planChannels({ goal: goal.outcome, records: modelNarrativeInformationOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind writeAccessibleTextOutput to writeAccessibleText({ goal: goal.outcome, records: planChannelsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind validateSynchronizationOutput to validateSynchronization({ goal: goal.outcome, records: writeAccessibleTextOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind packageAccessibleMediaOutput to packageAccessibleMedia({ goal: goal.outcome, records: validateSynchronizationOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { modelNarrativeInformation: modelNarrativeInformationOutput, planChannels: planChannelsOutput, writeAccessibleText: writeAccessibleTextOutput, validateSynchronization: validateSynchronizationOutput, packageAccessibleMedia: packageAccessibleMediaOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...modelNarrativeInformationOutput.nextQuestions, ...planChannelsOutput.nextQuestions, ...writeAccessibleTextOutput.nextQuestions, ...validateSynchronizationOutput.nextQuestions, ...packageAccessibleMediaOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { modelNarrativeInformation: modelNarrativeInformationOutput, planChannels: planChannelsOutput, writeAccessibleText: writeAccessibleTextOutput, validateSynchronization: validateSynchronizationOutput, packageAccessibleMedia: packageAccessibleMediaOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show an edit shifts later scenes, retime affected segments using stable scene identities. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a visual clue matters only later, describe what is visible without revealing its future meaning. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show dialogue leaves no description gap, propose an approved alternative channel or edit review. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a speaker is off-screen, identify them only when supported by the source. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a delivery platform interprets captions differently, validate the actual target format and player. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
