---
description: Prepare an auditable Health Record Request Fulfillment Workbench proposal.
args:
  request: string
  records: HealthRecordRequest[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['interpretAuthorization', 'locateRecords', 'reviewContent', 'assembleFulfillment', 'verifyDelivery'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind interpretAuthorizationOutput to interpretAuthorization({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind locateRecordsOutput to locateRecords({ goal: goal.outcome, records: interpretAuthorizationOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind reviewContentOutput to reviewContent({ goal: goal.outcome, records: locateRecordsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind assembleFulfillmentOutput to assembleFulfillment({ goal: goal.outcome, records: reviewContentOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind verifyDeliveryOutput to verifyDelivery({ goal: goal.outcome, records: assembleFulfillmentOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { interpretAuthorization: interpretAuthorizationOutput, locateRecords: locateRecordsOutput, reviewContent: reviewContentOutput, assembleFulfillment: assembleFulfillmentOutput, verifyDelivery: verifyDeliveryOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...interpretAuthorizationOutput.nextQuestions, ...locateRecordsOutput.nextQuestions, ...reviewContentOutput.nextQuestions, ...assembleFulfillmentOutput.nextQuestions, ...verifyDeliveryOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { interpretAuthorization: interpretAuthorizationOutput, locateRecords: locateRecordsOutput, reviewContent: reviewContentOutput, assembleFulfillment: assembleFulfillmentOutput, verifyDelivery: verifyDeliveryOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show authorization expires while review is pending, revalidate before delivery. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a record contains information about another person, route targeted review rather than releasing the whole document automatically. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a request covers only one episode, prevent unrelated history from entering the packet. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show an export contains hidden metadata, sanitize and verify it after rendering. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a corrected record arrives after disclosure, create a controlled amendment and notification obligation. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
