---
description: Prepare an auditable Referral Packet Completeness Manager proposal.
args:
  request: string
  records: ReferralCase[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['interpretRequirements', 'matchPatientRecords', 'assessCompleteness', 'requestMaterial', 'releaseAndTrack'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind interpretRequirementsOutput to interpretRequirements({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind matchPatientRecordsOutput to matchPatientRecords({ goal: goal.outcome, records: interpretRequirementsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind assessCompletenessOutput to assessCompleteness({ goal: goal.outcome, records: matchPatientRecordsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind requestMaterialOutput to requestMaterial({ goal: goal.outcome, records: assessCompletenessOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind releaseAndTrackOutput to releaseAndTrack({ goal: goal.outcome, records: requestMaterialOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { interpretRequirements: interpretRequirementsOutput, matchPatientRecords: matchPatientRecordsOutput, assessCompleteness: assessCompletenessOutput, requestMaterial: requestMaterialOutput, releaseAndTrack: releaseAndTrackOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...interpretRequirementsOutput.nextQuestions, ...matchPatientRecordsOutput.nextQuestions, ...assessCompletenessOutput.nextQuestions, ...requestMaterialOutput.nextQuestions, ...releaseAndTrackOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { interpretRequirements: interpretRequirementsOutput, matchPatientRecords: matchPatientRecordsOutput, assessCompleteness: assessCompletenessOutput, requestMaterial: requestMaterialOutput, releaseAndTrack: releaseAndTrackOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show two patients share a name, require approved identity evidence before matching documents. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a receiving service changes its checklist, revalidate only affected pending packets. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a document is present but unsigned, retain the gap rather than marking complete. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a clinician changes priority, preserve the authoritative change without independently re-triaging. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a transfer acknowledgment is lost, reconcile recipient status before resending sensitive records. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
