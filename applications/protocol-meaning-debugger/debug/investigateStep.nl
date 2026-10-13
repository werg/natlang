---
description: Consume one probe and update hypotheses from returned semantic evidence.
kind: directory-reducer
args:
  state: InvestigationState
  context: InvestigationContext
uses: [probes, probes/choose]
returns: InvestigationState
---
1. If pendingSeedIds is empty, return state. Otherwise call probes(context, state), then
   choose(candidates, state.hypotheses). Consume the actual chosen ID. Preserve returned candidates in state.candidates, one per seed ID;
   retain their actual typed prediction contributions instead of rerunning unchanged trials at every step.
2. If choose returns null, settle remaining IDs as exhausted with the precise missing discrimination
   reason. Return empty pending IDs; never retry an unchanged experiment frontier.
3. Locate the chosen candidate and its raw seed. If inadmissible, record an explicit rejected outcome
   with prediction:null and localDecision:null; a rejected agenda item never fabricates a literal invocation.
4. Otherwise call interpretOutcome(context, state.hypotheses, candidate). Its returned local semantic
   judgment owns support/rejection/uncertainty updates for this executed toy trial.
5. Update hypothesis statuses using those returned IDs. A supported hypothesis keeps its witnesses;
   an inconclusive result does not reject a mechanism that the seed did not actually isolate.
6. Write the outcome under experiments/<seedId>.json in the private folder. Include materialized
   plans and receipts, the actual interpreted outcome and its LocalDecision, not only the final percent.
7. Return a new state with this ID removed from pendingSeedIds, appended once to usedSeedIds,
   outcome appended, cached candidates preserved and unresolved preserved. Every continuing step strictly reduces the seed count.
8. A disagreement between the semantic mechanism prediction and the exact service receipt remains
   a typed unresolved experiment finding. Do not rewrite its prediction after seeing the observation to claim success.
9. The root decides structural stopping from hypothesis settlement or frontier exhaustion. This stage
   performs no external experiment, no source mutation and no arbitrary time/step budget.
