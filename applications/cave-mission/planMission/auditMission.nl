---
description: Replay the chosen policy in every supported physical world and challenge its assumptions.
args:
  request: MissionRequest
  survey: SurveyEvidence
  worlds: WorldSet
  planning: PlanningResult
returns: AuditedMission
---
1. Check every witness against survey.documents by sourceId/path and exact quote occurrence. Keep
   a defect for any missing document or altered quote; quotation presence alone does not prove support.
2. If no policy exists, return the actual worlds and policy, empty replays, source coverage and its blocked findings.
   A failed planning result remains explicit and is not replaced by a separate hand-written route.
3. For each supported WorldModel, call replayBranch(world, request, survey, planning.policy).
   Consume its final state, resource totals, success flag and local assumption findings.
4. Verify that every world has one replay and every actual sensor answer follows one declared policy
   edge. Compare the maximum replay cost/count with the policy's claimed worst-case objective.
5. Check every successful terminal state returns to request.start carrying the sealed sample, with
   nonnegative energy and source-consistent rope state. A live robot-only return is not enough.
6. Challenge the minimax value against the replayed necessary action costs and observation need.
   Equal-looking landmarks and a one-cell sensor partition cannot justify discarding a supported world.
7. Return AuditedMission with actual worlds, policy, all replays, findings and sourceCoverage from
   survey.coverage. The graph and values are internal typed stage results used in this audit; the final
   certificate exposes the chosen policy and evidence coverage without duplicating the search space.
