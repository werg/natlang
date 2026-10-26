---
description: Prepare an auditable Public Service Application Navigator proposal.
args:
  request: string
  records: ServiceNavigation[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['interpretSituation', 'identifyPathways', 'chooseQuestions', 'assembleApplications', 'trackOutcomes'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind interpretSituationOutput to interpretSituation({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind identifyPathwaysOutput to identifyPathways({ goal: goal.outcome, records: interpretSituationOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind chooseQuestionsOutput to chooseQuestions({ goal: goal.outcome, records: identifyPathwaysOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind assembleApplicationsOutput to assembleApplications({ goal: goal.outcome, records: chooseQuestionsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind trackOutcomesOutput to trackOutcomes({ goal: goal.outcome, records: assembleApplicationsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { interpretSituation: interpretSituationOutput, identifyPathways: identifyPathwaysOutput, chooseQuestions: chooseQuestionsOutput, assembleApplications: assembleApplicationsOutput, trackOutcomes: trackOutcomesOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...interpretSituationOutput.nextQuestions, ...identifyPathwaysOutput.nextQuestions, ...chooseQuestionsOutput.nextQuestions, ...assembleApplicationsOutput.nextQuestions, ...trackOutcomesOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { interpretSituation: interpretSituationOutput, identifyPathways: identifyPathwaysOutput, chooseQuestions: chooseQuestionsOutput, assembleApplications: assembleApplicationsOutput, trackOutcomes: trackOutcomesOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a person lacks a standard document, follow supplied alternative-evidence routes rather than declaring them ineligible. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a service rule changes mid-application, revalidate the affected pathway and explain the change. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a form cannot represent the person’s situation, route a truthful clarification instead of forcing a false answer. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a submission receipt is delayed, reconcile status before resubmitting. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a helper acts for the applicant, verify delegated consent and restrict access accordingly. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
