---
description: Prepare an auditable Volunteer Deployment Coordinator proposal.
args:
  request: string
  records: VolunteerMission[]
  evidence: EvidenceItem[]
  policyCandidates: EvidenceItem[]
  expectedCoverage: string[]
  recipient: string
  committedSnapshot: string[]
  sourceDelta: EvidenceItem[]
returns: RunProposal
---
1. Bind goal to the result of frameRequest({ request: request }). Bind policy to selectPolicy({ goal: goal, candidates: policyCandidates }). If selected authority is unresolved, keep that fact as a blocker; do not guess which policy controls.
2. Bind graph to formTaskGraph({ goal: goal, stages: ['interpretMission', 'interpretProfiles', 'constructTeams', 'prepareBriefings', 'replanAssignments'] }). Bind evidencePlan to planEvidence({ graph: graph, catalog: evidence }). Keep omitted partitions and missing prerequisites visible.
3. Bind interpretMissionOutput to interpretMission({ goal: goal.outcome, records: records, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
4. Bind interpretProfilesOutput to interpretProfiles({ goal: goal.outcome, records: interpretMissionOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
5. Bind constructTeamsOutput to constructTeams({ goal: goal.outcome, records: interpretProfilesOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
6. Bind prepareBriefingsOutput to prepareBriefings({ goal: goal.outcome, records: constructTeamsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
7. Bind replanAssignmentsOutput to replanAssignments({ goal: goal.outcome, records: prepareBriefingsOutput.accepted, evidence: evidence }). Keep its accepted domain records, unresolved candidates, evidence references, questions, and gate results in the checkpoint.
8. Bind collection to reconcileCollection({ stageOutputs: { interpretMission: interpretMissionOutput, interpretProfiles: interpretProfilesOutput, constructTeams: constructTeamsOutput, prepareBriefings: prepareBriefingsOutput, replanAssignments: replanAssignmentsOutput }, expectedCoverage: expectedCoverage }). Bind questions to resolveQuestions({ blockers: [...policy.unresolvedAuthority, ...interpretMissionOutput.nextQuestions, ...interpretProfilesOutput.nextQuestions, ...constructTeamsOutput.nextQuestions, ...prepareBriefingsOutput.nextQuestions, ...replanAssignmentsOutput.nextQuestions, ...collection.uncovered], replies: [] }). If a gate failed, bind a bounded repair proposal with proposeRevision using the exact finding and affected artifact; never weaken a gate or discard a qualified result.
9. Bind packet to prepareReview({ stageOutputs: { interpretMission: interpretMissionOutput, interpretProfiles: interpretProfilesOutput, constructTeams: constructTeamsOutput, prepareBriefings: prepareBriefingsOutput, replanAssignments: replanAssignmentsOutput }, finding: collection, questions: questions }). Bind view to composeView({ packet: packet, recipient: recipient }). Bind invalidation to assessDelta({ committedSnapshot: committedSnapshot, sourceDelta: sourceDelta, stageOutputs: packet.stageOutputs }).
10. If supplied facts show a volunteer withdraws at short notice, recompute capability coverage, not merely headcount. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
11. If supplied facts show a qualification expires, remove only assignments requiring it and notify the coordinator. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
12. If supplied facts show a role changes from remote to on-site, obtain fresh consent and check practical constraints. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
13. If supplied facts show a team member needs private access information, disclose it only for the accepted assignment window. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
14. If supplied facts show a trainee is assigned, preserve the required supervisor relationship. Keep this as a source-grounded proposal, preserve the relevant evidence link, and do not imply that the host has committed or dispatched it.
15. Return a RunProposal whose stageOutputs contains all five typed StageResult values, view is the recipient-authorized ViewModel, checks include invalidation.affectedArtifacts, invalidation.unresolvedImpact, and collection findings, and coverage names searched and omitted partitions. Mark it ready-for-review only when required gates pass and no decisive blocker remains; otherwise mark it blocked and name the missing evidence. The output is a proposal: policy approval, public claims, persistence, and external commitments remain with accountable humans and exact host services.
