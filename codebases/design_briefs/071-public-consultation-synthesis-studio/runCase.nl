---
description: Prepare an auditable Public Consultation Synthesis Studio proposal.
args:
  request: string
  records: Consultation[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['extractArguments', 'linkCampaigns', 'mapPositions', 'formResponses', 'verifySynthesis'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind extractArgumentsOutput to extractArguments({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind linkCampaignsOutput to linkCampaigns({ goal: goal.outcome, records: extractArgumentsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind mapPositionsOutput to mapPositions({ goal: goal.outcome, records: linkCampaignsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind formResponsesOutput to formResponses({ goal: goal.outcome, records: mapPositionsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind verifySynthesisOutput to verifySynthesis({ goal: goal.outcome, records: formResponsesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { extractArguments: extractArgumentsOutput, linkCampaigns: linkCampaignsOutput, mapPositions: mapPositionsOutput, formResponses: formResponsesOutput, verifySynthesis: verifySynthesisOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...extractArgumentsOutput.nextQuestions, ...linkCampaignsOutput.nextQuestions, ...mapPositionsOutput.nextQuestions, ...formResponsesOutput.nextQuestions, ...verifySynthesisOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { extractArguments: extractArgumentsOutput, linkCampaigns: linkCampaignsOutput, mapPositions: mapPositionsOutput, formResponses: formResponsesOutput, verifySynthesis: verifySynthesisOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a campaign produces many identical submissions, report participation volume and shared argument structure separately. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a submission supports one part and opposes another, preserve both scoped positions. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a quoted example reveals a private circumstance, anonymize with consent-aware review. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a late submission is accepted under an exception, record its inclusion basis. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a report is challenged for omission, trace the concern back to coverage and revise transparently. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
