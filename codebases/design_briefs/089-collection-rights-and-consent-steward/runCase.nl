---
description: Prepare an auditable Collection Rights And Consent Steward proposal.
args:
  request: string
  records: CollectionUse[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['extractPermissions', 'interpretUse', 'assessCompatibility', 'requestClearance', 'maintainUseRegister'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind extractPermissionsOutput to extractPermissions({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind interpretUseOutput to interpretUse({ goal: goal.outcome, records: extractPermissionsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind assessCompatibilityOutput to assessCompatibility({ goal: goal.outcome, records: interpretUseOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind requestClearanceOutput to requestClearance({ goal: goal.outcome, records: assessCompatibilityOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind maintainUseRegisterOutput to maintainUseRegister({ goal: goal.outcome, records: requestClearanceOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { extractPermissions: extractPermissionsOutput, interpretUse: interpretUseOutput, assessCompatibility: assessCompatibilityOutput, requestClearance: requestClearanceOutput, maintainUseRegister: maintainUseRegisterOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...extractPermissionsOutput.nextQuestions, ...interpretUseOutput.nextQuestions, ...assessCompatibilityOutput.nextQuestions, ...requestClearanceOutput.nextQuestions, ...maintainUseRegisterOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { extractPermissions: extractPermissionsOutput, interpretUse: interpretUseOutput, assessCompatibility: assessCompatibilityOutput, requestClearance: requestClearanceOutput, maintainUseRegister: maintainUseRegisterOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a donor owns the object but not reproduction rights, require the appropriate separate clearance. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a license allows print but not online use, prevent automatic web publication. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show consent is time-limited, schedule expiry review and removal obligations. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a derivative combines several items, check every relevant permission. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show attribution changes after a name correction, update affected publications through a traceable process. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
