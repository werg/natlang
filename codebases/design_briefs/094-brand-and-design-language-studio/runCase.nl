---
description: Prepare an auditable Brand And Design Language Studio proposal.
args:
  request: string
  records: BrandProject[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['recoverBrandConstraints', 'generateDirections', 'constructSystem', 'applyChannels', 'reviewCoherence'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind recoverBrandConstraintsOutput to recoverBrandConstraints({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind generateDirectionsOutput to generateDirections({ goal: goal.outcome, records: recoverBrandConstraintsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind constructSystemOutput to constructSystem({ goal: goal.outcome, records: generateDirectionsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind applyChannelsOutput to applyChannels({ goal: goal.outcome, records: constructSystemOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind reviewCoherenceOutput to reviewCoherence({ goal: goal.outcome, records: applyChannelsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { recoverBrandConstraints: recoverBrandConstraintsOutput, generateDirections: generateDirectionsOutput, constructSystem: constructSystemOutput, applyChannels: applyChannelsOutput, reviewCoherence: reviewCoherenceOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...recoverBrandConstraintsOutput.nextQuestions, ...generateDirectionsOutput.nextQuestions, ...constructSystemOutput.nextQuestions, ...applyChannelsOutput.nextQuestions, ...reviewCoherenceOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { recoverBrandConstraints: recoverBrandConstraintsOutput, generateDirections: generateDirectionsOutput, constructSystem: constructSystemOutput, applyChannels: applyChannelsOutput, reviewCoherence: reviewCoherenceOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a slogan implies a capability the product lacks, reject or qualify it despite creative appeal. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a name may conflict with existing rights, route clearance rather than claiming availability. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show error messages inherit promotional tone inappropriately, define context-specific voice rules. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show the product changes, invalidate affected claims and assets, not the entire visual direction. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a client approves one asset but not the system, preserve the narrower approval scope. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
