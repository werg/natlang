---
description: Prepare an auditable Multilingual Terminology Negotiation Workbench proposal.
args:
  request: string
  records: TerminologyProject[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['extractSenses', 'alignConcepts', 'constructTermOptions', 'negotiateDecision', 'propagateTerminology'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind extractSensesOutput to extractSenses({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind alignConceptsOutput to alignConcepts({ goal: goal.outcome, records: extractSensesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind constructTermOptionsOutput to constructTermOptions({ goal: goal.outcome, records: alignConceptsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind negotiateDecisionOutput to negotiateDecision({ goal: goal.outcome, records: constructTermOptionsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind propagateTerminologyOutput to propagateTerminology({ goal: goal.outcome, records: negotiateDecisionOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { extractSenses: extractSensesOutput, alignConcepts: alignConceptsOutput, constructTermOptions: constructTermOptionsOutput, negotiateDecision: negotiateDecisionOutput, propagateTerminology: propagateTerminologyOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...extractSensesOutput.nextQuestions, ...alignConceptsOutput.nextQuestions, ...constructTermOptionsOutput.nextQuestions, ...negotiateDecisionOutput.nextQuestions, ...propagateTerminologyOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { extractSenses: extractSensesOutput, alignConcepts: alignConceptsOutput, constructTermOptions: constructTermOptionsOutput, negotiateDecision: negotiateDecisionOutput, propagateTerminology: propagateTerminologyOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a term has two meanings in one product, bind terminology to concept and context. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a regional variant is required, preserve its scope rather than declaring it inconsistent. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a term changes grammatical form, generate context-aware inflections and review them. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show experts disagree on the concept boundary, keep the issue open instead of choosing the most common word. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a glossary decision affects released content, stage a versioned migration and avoid mixed terminology in one workflow. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
