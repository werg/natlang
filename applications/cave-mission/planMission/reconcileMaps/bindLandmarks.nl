---
description: Use scoped description judgments to separate genuine anchors from repeated visual markers.
args:
  survey: SurveyEvidence
returns: LandmarkPartition
---
1. Enumerate candidate pairs from shared descriptions and explicit cross-survey references. Include
   repeated marks and unique anchors so the identity distinction is judged from source evidence.
2. For each pair, gather its two original LandmarkMentions and only governing SourceClauses about
   reuse, uniqueness and frame ownership. Keep those original clause IDs and witnesses in parent scope.
3. In eval declare local LandmarkMeaning as {relation:'same'|'different'|'unresolved'; sharedFrameAnchor:boolean; reason:string;
   evidence:{clauseId:string; exactQuote:string}[]}. Create nl.with<LandmarkMeaning> capturing the
   governing clauses. Directly await it on {left:leftMention,right:rightMention} for this changing pair.
4. The literal judges same, different or unresolved physical-place identity, sharedFrameAnchor and
   the source-based reason. It returns nonempty exact quoted evidence keyed by a supplied clauseId.
   A repeated appearance alone leaves identity unresolved; an explicit unique shared anchor can establish it.
5. Consume the actual returned meaning. For every evidence item, find its original governing clause
   by exact clauseId. Require a nonempty exactQuote occurring within at least one original clause witness
   quote. Bind each result Witness.sourceId and path from that matched original witness in the parent.
6. Exact ID/substring checks are provenance mechanism only. If evidence is missing or a quote fails,
   fail this stage with the affected source identity and quotation-check issue so its caller preserves
   the failed contribution. Consume an identity decision only after its nonempty witness set passes.
7. Construct LandmarkDecision.leftId/rightId from the original pair and its relation/sharedFrameAnchor
   from the actual literal result. Parent-owned provenance is attached after exact validation.
8. Consume same-place anchor decisions into fixedBindings. Keep unresolved pairs available for
   conditional resolution inside each transform; they supply no global equality constraint.
9. Compute transitive identity closure only when every join is source-supported. Keep an explicitly
   different pair distinct even if another chain of similar appearances would connect it.
10. Return actual decisions, fixedBindings and unresolvedPairs. The next stage consumes these typed
    distinctions, preserving unresolved actual reflection while evaluating each possible world separately.
