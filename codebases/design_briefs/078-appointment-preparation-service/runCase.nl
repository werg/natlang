---
description: Prepare an auditable Appointment Preparation Service proposal.
args:
  request: string
  records: AppointmentPreparation[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['recoverInstructions', 'interpretCircumstances', 'composeChecklist', 'routeQuestions', 'trackAcknowledgment'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind recoverInstructionsOutput to recoverInstructions({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind interpretCircumstancesOutput to interpretCircumstances({ goal: goal.outcome, records: recoverInstructionsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind composeChecklistOutput to composeChecklist({ goal: goal.outcome, records: interpretCircumstancesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind routeQuestionsOutput to routeQuestions({ goal: goal.outcome, records: composeChecklistOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind trackAcknowledgmentOutput to trackAcknowledgment({ goal: goal.outcome, records: routeQuestionsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { recoverInstructions: recoverInstructionsOutput, interpretCircumstances: interpretCircumstancesOutput, composeChecklist: composeChecklistOutput, routeQuestions: routeQuestionsOutput, trackAcknowledgment: trackAcknowledgmentOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...recoverInstructionsOutput.nextQuestions, ...interpretCircumstancesOutput.nextQuestions, ...composeChecklistOutput.nextQuestions, ...routeQuestionsOutput.nextQuestions, ...trackAcknowledgmentOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { recoverInstructions: recoverInstructionsOutput, interpretCircumstances: interpretCircumstancesOutput, composeChecklist: composeChecklistOutput, routeQuestions: routeQuestionsOutput, trackAcknowledgment: trackAcknowledgmentOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show an appointment time changes, recompute exact relative timings and reissue the affected checklist. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a translated phrase changes an exception, block release pending fidelity review. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a patient reports conflicting instructions, route to staff rather than choosing one. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a representative receives messages, verify current consent and scope. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a message is undelivered, use approved fallback contact rather than assuming receipt. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
