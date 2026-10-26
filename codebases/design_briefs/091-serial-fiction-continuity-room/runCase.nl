---
description: Prepare an auditable Serial Fiction Continuity Room proposal.
args:
  request: string
  records: StorySeries[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['recoverCanon', 'trackArcs', 'assessProposals', 'generateRepairs', 'verifyEdition'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind recoverCanonOutput to recoverCanon({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind trackArcsOutput to trackArcs({ goal: goal.outcome, records: recoverCanonOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind assessProposalsOutput to assessProposals({ goal: goal.outcome, records: trackArcsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind generateRepairsOutput to generateRepairs({ goal: goal.outcome, records: assessProposalsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind verifyEditionOutput to verifyEdition({ goal: goal.outcome, records: generateRepairsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { recoverCanon: recoverCanonOutput, trackArcs: trackArcsOutput, assessProposals: assessProposalsOutput, generateRepairs: generateRepairsOutput, verifyEdition: verifyEditionOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...recoverCanonOutput.nextQuestions, ...trackArcsOutput.nextQuestions, ...assessProposalsOutput.nextQuestions, ...generateRepairsOutput.nextQuestions, ...verifyEditionOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { recoverCanon: recoverCanonOutput, trackArcs: trackArcsOutput, assessProposals: assessProposalsOutput, generateRepairs: generateRepairsOutput, verifyEdition: verifyEditionOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a character learns a secret in one draft but not another, bind knowledge to the approved edition. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show an unreliable narrator states a falsehood, preserve it as narration, not canon. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a late twist changes earlier scenes, show the full seeding and revision footprint. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a minor character returns after many installments, retrieve their established voice and history. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a deliberate contradiction is part of the plot, record the author-approved rationale and reveal plan. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
