---
description: Prepare an auditable Culinary Concept Development Kitchen proposal.
args:
  request: string
  records: MenuProject[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['interpretBrief', 'generateDishes', 'constructMenu', 'planAndCost', 'reviewTastings'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind interpretBriefOutput to interpretBrief({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind generateDishesOutput to generateDishes({ goal: goal.outcome, records: interpretBriefOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind constructMenuOutput to constructMenu({ goal: goal.outcome, records: generateDishesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind planAndCostOutput to planAndCost({ goal: goal.outcome, records: constructMenuOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind reviewTastingsOutput to reviewTastings({ goal: goal.outcome, records: planAndCostOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { interpretBrief: interpretBriefOutput, generateDishes: generateDishesOutput, constructMenu: constructMenuOutput, planAndCost: planAndCostOutput, reviewTastings: reviewTastingsOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...interpretBriefOutput.nextQuestions, ...generateDishesOutput.nextQuestions, ...constructMenuOutput.nextQuestions, ...planAndCostOutput.nextQuestions, ...reviewTastingsOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { interpretBrief: interpretBriefOutput, generateDishes: generateDishesOutput, constructMenu: constructMenuOutput, planAndCost: planAndCostOutput, reviewTastings: reviewTastingsOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a supplier substitutes an ingredient, recheck verified allergen and dietary data before use. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show yield differs from the estimate, update exact batch quantities and cost. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show two dishes need the same equipment simultaneously, replan service rather than assuming unlimited capacity. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a tasting change affects a shared sauce, propagate it to every dish and disclosure. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a menu label claims allergen-free preparation, require the venue’s verified controls and qualified approval. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
