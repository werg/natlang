---
description: Prepare an auditable Care Transition Logistics Coordinator proposal.
args:
  request: string
  records: CareTransition[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['extractLogistics', 'mapParticipants', 'constructSequence', 'resolveConflicts', 'verifyReadiness'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind extractLogisticsOutput to extractLogistics({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind mapParticipantsOutput to mapParticipants({ goal: goal.outcome, records: extractLogisticsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind constructSequenceOutput to constructSequence({ goal: goal.outcome, records: mapParticipantsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind resolveConflictsOutput to resolveConflicts({ goal: goal.outcome, records: constructSequenceOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind verifyReadinessOutput to verifyReadiness({ goal: goal.outcome, records: resolveConflictsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { extractLogistics: extractLogisticsOutput, mapParticipants: mapParticipantsOutput, constructSequence: constructSequenceOutput, resolveConflicts: resolveConflictsOutput, verifyReadiness: verifyReadinessOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...extractLogisticsOutput.nextQuestions, ...mapParticipantsOutput.nextQuestions, ...constructSequenceOutput.nextQuestions, ...resolveConflictsOutput.nextQuestions, ...verifyReadinessOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { extractLogistics: extractLogisticsOutput, mapParticipants: mapParticipantsOutput, constructSequence: constructSequenceOutput, resolveConflicts: resolveConflictsOutput, verifyReadiness: verifyReadinessOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show equipment is booked but not delivered, keep the dependent transition step blocked. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a caregiver cannot accept the role, escalate responsibility rather than assuming availability. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show transport timing changes, recheck downstream appointments and access windows. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a proposed substitution affects clinical requirements, obtain clinician review before proceeding. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a receiving organization has not acknowledged the plan, preserve the unresolved handoff. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
