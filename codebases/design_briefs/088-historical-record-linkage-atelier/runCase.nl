---
description: Prepare an auditable Historical Record Linkage Atelier proposal.
args:
  request: string
  records: HistoricalLinkage[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['normalizeRecords', 'generateLinks', 'judgeIdentity', 'reconcileNetworks', 'prepareResearchView'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind normalizeRecordsOutput to normalizeRecords({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind generateLinksOutput to generateLinks({ goal: goal.outcome, records: normalizeRecordsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind judgeIdentityOutput to judgeIdentity({ goal: goal.outcome, records: generateLinksOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind reconcileNetworksOutput to reconcileNetworks({ goal: goal.outcome, records: judgeIdentityOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind prepareResearchViewOutput to prepareResearchView({ goal: goal.outcome, records: reconcileNetworksOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { normalizeRecords: normalizeRecordsOutput, generateLinks: generateLinksOutput, judgeIdentity: judgeIdentityOutput, reconcileNetworks: reconcileNetworksOutput, prepareResearchView: prepareResearchViewOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...normalizeRecordsOutput.nextQuestions, ...generateLinksOutput.nextQuestions, ...judgeIdentityOutput.nextQuestions, ...reconcileNetworksOutput.nextQuestions, ...prepareResearchViewOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { normalizeRecords: normalizeRecordsOutput, generateLinks: generateLinksOutput, judgeIdentity: judgeIdentityOutput, reconcileNetworks: reconcileNetworksOutput, prepareResearchView: prepareResearchViewOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a place changes name or boundary, preserve historical geography rather than forcing a modern identity. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a date is approximate, use intervals in chronology checks. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a family reuses names across generations, require relationship and timing evidence. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a transcription is corrected, invalidate dependent links selectively. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a researcher merges entities prematurely, support a reversible split with downstream impact tracking. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
