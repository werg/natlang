---
description: Prepare an auditable Community Agreement Mediator proposal.
args:
  request: string
  records: CommunityDispute[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['recoverAgreements', 'mapPositions', 'identifyOptions', 'designTrial', 'reviewOutcome'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind recoverAgreementsOutput to recoverAgreements({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind mapPositionsOutput to mapPositions({ goal: goal.outcome, records: recoverAgreementsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind identifyOptionsOutput to identifyOptions({ goal: goal.outcome, records: mapPositionsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind designTrialOutput to designTrial({ goal: goal.outcome, records: identifyOptionsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind reviewOutcomeOutput to reviewOutcome({ goal: goal.outcome, records: designTrialOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { recoverAgreements: recoverAgreementsOutput, mapPositions: mapPositionsOutput, identifyOptions: identifyOptionsOutput, designTrial: designTrialOutput, reviewOutcome: reviewOutcomeOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...recoverAgreementsOutput.nextQuestions, ...mapPositionsOutput.nextQuestions, ...identifyOptionsOutput.nextQuestions, ...designTrialOutput.nextQuestions, ...reviewOutcomeOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { recoverAgreements: recoverAgreementsOutput, mapPositions: mapPositionsOutput, identifyOptions: identifyOptionsOutput, designTrial: designTrialOutput, reviewOutcome: reviewOutcomeOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show one participant does not respond, keep consent pending rather than assuming acceptance. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a prior agreement is disputed, preserve competing accounts and seek evidence. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a trial works for some but not all, report divergent outcomes instead of averaging them away. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a participant withdraws consent, apply the agreed exit process promptly. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a proposal concerns legal rights or immediate safety, route it to appropriate qualified support rather than mediating beyond scope. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
