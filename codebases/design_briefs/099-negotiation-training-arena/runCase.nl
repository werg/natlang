---
description: Prepare an auditable Negotiation Training Arena proposal.
args:
  request: string
  records: NegotiationExercise[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['authorNegotiation', 'prepareRoles', 'interpretOffers', 'runCounterparts', 'debriefNegotiation'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind authorNegotiationOutput to authorNegotiation({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind prepareRolesOutput to prepareRoles({ goal: goal.outcome, records: authorNegotiationOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind interpretOffersOutput to interpretOffers({ goal: goal.outcome, records: prepareRolesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind runCounterpartsOutput to runCounterparts({ goal: goal.outcome, records: interpretOffersOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind debriefNegotiationOutput to debriefNegotiation({ goal: goal.outcome, records: runCounterpartsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { authorNegotiation: authorNegotiationOutput, prepareRoles: prepareRolesOutput, interpretOffers: interpretOffersOutput, runCounterparts: runCounterpartsOutput, debriefNegotiation: debriefNegotiationOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...authorNegotiationOutput.nextQuestions, ...prepareRolesOutput.nextQuestions, ...interpretOffersOutput.nextQuestions, ...runCounterpartsOutput.nextQuestions, ...debriefNegotiationOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { authorNegotiation: authorNegotiationOutput, prepareRoles: prepareRolesOutput, interpretOffers: interpretOffersOutput, runCounterparts: runCounterpartsOutput, debriefNegotiation: debriefNegotiationOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a participant makes an ambiguous conditional offer, clarify before updating commitments. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a role exceeds its authority, reject the simulated commitment and retain the learning event. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show an instructor changes difficulty mid-session, preserve the original hidden world or create an explicit new episode. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a participant withdraws an offer, apply the scenario’s exact acceptance timing rules. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show debrief materials reveal private scores, release them only under the exercise’s approved debrief policy. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
