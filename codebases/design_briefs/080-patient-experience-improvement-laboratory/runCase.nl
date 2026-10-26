---
description: Prepare an auditable Patient Experience Improvement Laboratory proposal.
args:
  request: string
  records: ExperienceStudy[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['extractExperiences', 'linkJourneyEvents', 'formMechanisms', 'designImprovements', 'reviewResults'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind extractExperiencesOutput to extractExperiences({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind linkJourneyEventsOutput to linkJourneyEvents({ goal: goal.outcome, records: extractExperiencesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind formMechanismsOutput to formMechanisms({ goal: goal.outcome, records: linkJourneyEventsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind designImprovementsOutput to designImprovements({ goal: goal.outcome, records: formMechanismsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind reviewResultsOutput to reviewResults({ goal: goal.outcome, records: designImprovementsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { extractExperiences: extractExperiencesOutput, linkJourneyEvents: linkJourneyEventsOutput, formMechanisms: formMechanismsOutput, designImprovements: designImprovementsOutput, reviewResults: reviewResultsOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...extractExperiencesOutput.nextQuestions, ...linkJourneyEventsOutput.nextQuestions, ...formMechanismsOutput.nextQuestions, ...designImprovementsOutput.nextQuestions, ...reviewResultsOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { extractExperiences: extractExperiencesOutput, linkJourneyEvents: linkJourneyEventsOutput, formMechanisms: formMechanismsOutput, designImprovements: designImprovementsOutput, reviewResults: reviewResultsOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a complaint includes clinical concerns, route them to the appropriate clinical process without interpreting them here. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show call volume falls because a channel closes, do not report that as improved access. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a small group faces severe barriers, preserve its experience despite aggregate improvement. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show feedback consent is withdrawn, remove attributable material and reassess derivatives. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show an experiment shifts work to another team, measure the downstream burden before adoption. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
