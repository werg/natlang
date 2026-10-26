---
description: Prepare an auditable Localization Release Manager proposal.
args:
  request: string
  records: LocalizationRelease[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['segmentWithContext', 'resolveTerminology', 'transformContent', 'validateLocale', 'assembleLocaleRelease'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind segmentWithContextOutput to segmentWithContext({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind resolveTerminologyOutput to resolveTerminology({ goal: goal.outcome, records: segmentWithContextOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind transformContentOutput to transformContent({ goal: goal.outcome, records: resolveTerminologyOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind validateLocaleOutput to validateLocale({ goal: goal.outcome, records: transformContentOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind assembleLocaleReleaseOutput to assembleLocaleRelease({ goal: goal.outcome, records: validateLocaleOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { segmentWithContext: segmentWithContextOutput, resolveTerminology: resolveTerminologyOutput, transformContent: transformContentOutput, validateLocale: validateLocaleOutput, assembleLocaleRelease: assembleLocaleReleaseOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...segmentWithContextOutput.nextQuestions, ...resolveTerminologyOutput.nextQuestions, ...transformContentOutput.nextQuestions, ...validateLocaleOutput.nextQuestions, ...assembleLocaleReleaseOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { segmentWithContext: segmentWithContextOutput, resolveTerminology: resolveTerminologyOutput, transformContent: transformContentOutput, validateLocale: validateLocaleOutput, assembleLocaleRelease: assembleLocaleReleaseOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show the same word labels both an action and a status, keep separate context keys. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a placeholder moves grammatically, allow reordering while enforcing exact identity and multiplicity. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a source edit changes only punctuation, assess semantic impact before invalidating all translations. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show one locale is delayed, apply an explicit fallback rather than mixing old and new critical instructions. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a glossary term changes, re-evaluate only segments using the affected concept and sense. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
