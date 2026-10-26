---
description: Prepare an auditable Oral History Reconciliation Archive proposal.
args:
  request: string
  records: OralHistoryProject[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['segmentAccounts', 'resolveReferences', 'reconcileEvents', 'formFollowups', 'publishArchive'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind segmentAccountsOutput to segmentAccounts({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind resolveReferencesOutput to resolveReferences({ goal: goal.outcome, records: segmentAccountsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind reconcileEventsOutput to reconcileEvents({ goal: goal.outcome, records: resolveReferencesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind formFollowupsOutput to formFollowups({ goal: goal.outcome, records: reconcileEventsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind publishArchiveOutput to publishArchive({ goal: goal.outcome, records: formFollowupsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { segmentAccounts: segmentAccountsOutput, resolveReferences: resolveReferencesOutput, reconcileEvents: reconcileEventsOutput, formFollowups: formFollowupsOutput, publishArchive: publishArchiveOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...segmentAccountsOutput.nextQuestions, ...resolveReferencesOutput.nextQuestions, ...reconcileEventsOutput.nextQuestions, ...formFollowupsOutput.nextQuestions, ...publishArchiveOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { segmentAccounts: segmentAccountsOutput, resolveReferences: resolveReferencesOutput, reconcileEvents: reconcileEventsOutput, formFollowups: formFollowupsOutput, publishArchive: publishArchiveOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a participant withdraws one anecdote, invalidate its quotations and derived summaries without deleting unrelated consented material. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show two speakers use different dates for one event, preserve both with context. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a nickname refers to several people, retain candidate identities. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a transcript correction changes attribution, re-evaluate dependent narratives. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a restricted interview supports a public summary, ensure the summary does not reveal protected details indirectly. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
