---
description: Extend each actual partial edition by one location and consume compatibility judgments.
args:
  state: HypothesisFrontier
returns: HypothesisFrontier
---
1. Return state when pendingBundles is empty. Bind bundle to its first actual item and beforeHypotheses
   to the current partial editions. Initialize nextHypotheses empty; copy rejected/unresolved collections.
2. If bundle.options is empty, record this original event location as incomplete for every actual
   partial hypothesis. Consume the bundle with no falsely complete extension; preserve prior rejection evidence.
3. Otherwise visit every pair of an actual partial hypothesis and one actual bundle option. Preserve
   that pair as immutable input and directly await checkClaimConsistency(hypothesis, option).
4. For compatible=true construct a fresh extended hypothesis. Append the actual option once, merge
   agreeing original claim assignments, and retain both prior and new exact source witnesses. No sibling
   extension may mutate this parent's choices or assignment map.
5. For compatible=false append RejectedHypothesis with this actual partial hypothesisId/eventId,
   priorOptionIds from its actual choices, priorAssignments, incomingOptionId and incomingAssignments
   copied from this actual option, plus the returned contradiction witnesses. The rejection belongs to that combination, not every option
   sharing one pitch or manuscript witness.
6. For compatible=null append the actual unresolved premise and pair identity. Keep its status distinct
   from a supported edition; missing source knowledge cannot be repaired by choosing the first option.
7. After all pairs, compare extended hypotheses by complete chosen event fields plus all source-level
   assignments. Coalesce behavior-equivalent duplicate serializations only; retain materially distinct
   editorial explanations even when one local event happens to have the same final numerical rank.
8. Assign reproducible generated hypothesis IDs to the actual retained extensions. IDs are reporting
   labels and cannot decide which branch survives. Preserve every earlier rejected/unresolved record.
9. Remove exactly this original bundle and return nextHypotheses plus findings. The measure decreases
   with remaining bundles even when the number of live hypotheses grows; no arbitrary beam limit applies.
