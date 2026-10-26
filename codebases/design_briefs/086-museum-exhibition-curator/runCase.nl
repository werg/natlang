---
description: Prepare an auditable Museum Exhibition Curator proposal.
args:
  request: string
  records: ExhibitionProject[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['interpretCollection', 'formNarratives', 'designVisitorPaths', 'writeInterpretation', 'qualifyExhibition'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind interpretCollectionOutput to interpretCollection({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind formNarrativesOutput to formNarratives({ goal: goal.outcome, records: interpretCollectionOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind designVisitorPathsOutput to designVisitorPaths({ goal: goal.outcome, records: formNarrativesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind writeInterpretationOutput to writeInterpretation({ goal: goal.outcome, records: designVisitorPathsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind qualifyExhibitionOutput to qualifyExhibition({ goal: goal.outcome, records: writeInterpretationOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { interpretCollection: interpretCollectionOutput, formNarratives: formNarrativesOutput, designVisitorPaths: designVisitorPathsOutput, writeInterpretation: writeInterpretationOutput, qualifyExhibition: qualifyExhibitionOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...interpretCollectionOutput.nextQuestions, ...formNarrativesOutput.nextQuestions, ...designVisitorPathsOutput.nextQuestions, ...writeInterpretationOutput.nextQuestions, ...qualifyExhibitionOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { interpretCollection: interpretCollectionOutput, formNarratives: formNarrativesOutput, designVisitorPaths: designVisitorPathsOutput, writeInterpretation: writeInterpretationOutput, qualifyExhibition: qualifyExhibitionOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show an artifact loan is withdrawn, revise the argument rather than silently substituting a weaker example. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a label exceeds the available space, shorten without removing a crucial qualification. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a community restriction limits display, enforce it across images, labels, and online previews. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a conservation condition changes, re-evaluate duration and placement. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a visitor skips a zone, ensure later labels do not require inaccessible prior context. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
