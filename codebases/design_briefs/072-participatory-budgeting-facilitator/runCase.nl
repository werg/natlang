---
description: Prepare an auditable Participatory Budgeting Facilitator proposal.
args:
  request: string
  records: BudgetRound[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['interpretProposals', 'assessOverlap', 'qualifyFeasibility', 'constructPortfolios', 'prepareDeliberation'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind interpretProposalsOutput to interpretProposals({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind assessOverlapOutput to assessOverlap({ goal: goal.outcome, records: interpretProposalsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind qualifyFeasibilityOutput to qualifyFeasibility({ goal: goal.outcome, records: assessOverlapOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind constructPortfoliosOutput to constructPortfolios({ goal: goal.outcome, records: qualifyFeasibilityOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind prepareDeliberationOutput to prepareDeliberation({ goal: goal.outcome, records: constructPortfoliosOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { interpretProposals: interpretProposalsOutput, assessOverlap: assessOverlapOutput, qualifyFeasibility: qualifyFeasibilityOutput, constructPortfolios: constructPortfoliosOutput, prepareDeliberation: prepareDeliberationOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...interpretProposalsOutput.nextQuestions, ...assessOverlapOutput.nextQuestions, ...qualifyFeasibilityOutput.nextQuestions, ...constructPortfoliosOutput.nextQuestions, ...prepareDeliberationOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { interpretProposals: interpretProposalsOutput, assessOverlap: assessOverlapOutput, qualifyFeasibility: qualifyFeasibilityOutput, constructPortfolios: constructPortfoliosOutput, prepareDeliberation: prepareDeliberationOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show two proposals share enabling work, count its cost once and expose the dependency. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a revised estimate exceeds the budget, reopen portfolio feasibility before voting or approval. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a merged proposal changes one group’s intent, require explicit proposer review. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a project is location-sensitive, show the relevant supplied constraints without profiling residents. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a selected project changes during delivery, record the scope change and applicable community approval rule. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
