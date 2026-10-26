---
description: Prepare an auditable Cross Cultural Training Adaptation Studio proposal.
args:
  request: string
  records: TrainingAdaptation[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['recoverLearningMechanism', 'assessContextFit', 'generateAlternatives', 'replayLearning', 'releaseTraining'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind recoverLearningMechanismOutput to recoverLearningMechanism({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind assessContextFitOutput to assessContextFit({ goal: goal.outcome, records: recoverLearningMechanismOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind generateAlternativesOutput to generateAlternatives({ goal: goal.outcome, records: assessContextFitOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind replayLearningOutput to replayLearning({ goal: goal.outcome, records: generateAlternativesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind releaseTrainingOutput to releaseTraining({ goal: goal.outcome, records: replayLearningOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { recoverLearningMechanism: recoverLearningMechanismOutput, assessContextFit: assessContextFitOutput, generateAlternatives: generateAlternativesOutput, replayLearning: replayLearningOutput, releaseTraining: releaseTrainingOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...recoverLearningMechanismOutput.nextQuestions, ...assessContextFitOutput.nextQuestions, ...generateAlternativesOutput.nextQuestions, ...replayLearningOutput.nextQuestions, ...releaseTrainingOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { recoverLearningMechanism: recoverLearningMechanismOutput, assessContextFit: assessContextFitOutput, generateAlternatives: generateAlternativesOutput, replayLearning: replayLearningOutput, releaseTraining: releaseTrainingOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show an analogy is unfamiliar but the concept is essential, replace the analogy, not the objective. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a reviewer disagrees with an audience assumption, preserve the disagreement and narrow the adaptation scope. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a scenario becomes easier after adaptation, restore the relevant decision complexity. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a local example could reinforce a stereotype, revise using supplied context rather than group generalization. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show the source curriculum changes, revalidate affected adapted mechanisms. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
