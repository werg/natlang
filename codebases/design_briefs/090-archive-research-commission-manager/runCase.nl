---
description: Prepare an auditable Archive Research Commission Manager proposal.
args:
  request: string
  records: ArchiveCommission[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['frameCommission', 'planSearch', 'followEvidence', 'reconcileFindings', 'assembleDossier'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind frameCommissionOutput to frameCommission({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind planSearchOutput to planSearch({ goal: goal.outcome, records: frameCommissionOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind followEvidenceOutput to followEvidence({ goal: goal.outcome, records: planSearchOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind reconcileFindingsOutput to reconcileFindings({ goal: goal.outcome, records: followEvidenceOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind assembleDossierOutput to assembleDossier({ goal: goal.outcome, records: reconcileFindingsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { frameCommission: frameCommissionOutput, planSearch: planSearchOutput, followEvidence: followEvidenceOutput, reconcileFindings: reconcileFindingsOutput, assembleDossier: assembleDossierOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...frameCommissionOutput.nextQuestions, ...planSearchOutput.nextQuestions, ...followEvidenceOutput.nextQuestions, ...reconcileFindingsOutput.nextQuestions, ...assembleDossierOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { frameCommission: frameCommissionOutput, planSearch: planSearchOutput, followEvidence: followEvidenceOutput, reconcileFindings: reconcileFindingsOutput, assembleDossier: assembleDossierOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a catalog entry points to missing material, record the lead and absence without inventing contents. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a collection is temporarily closed, preserve the coverage limitation. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show historical terminology differs from the request, maintain query variants and their results. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a promising lead exceeds budget, present a scoped extension decision instead of continuing silently. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a source cannot be reproduced publicly, provide an authorized citation or description rather than the restricted image. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
