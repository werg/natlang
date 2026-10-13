---
description: Use scoped description judgments to separate genuine anchors from repeated visual markers.
args:
  survey: SurveyEvidence
returns: LandmarkPartition
---
1. Enumerate candidate pairs suggested by shared descriptions or explicit cross-survey references.
   Include tempting repeated marks as well as explicitly unique anchors so their distinction is recorded.
2. For each pair, gather the two descriptions and only the clauses governing marker reuse, uniqueness
   or frame ownership. Keep the local labels and physical-place question explicit in the scoped context.
3. In eval declare a local LandmarkJudgment matching LandmarkDecision. Create nl.with<LandmarkJudgment>
   capturing those governing clauses, and directly await it on the changing pair of LandmarkMentions.
4. Ask the literal whether the descriptions establish same place, different places or unresolved
   identity; require a sharedFrameAnchor judgment and exact source witnesses. Mere similarity is not identity.
5. Consume same-place anchor results into fixedBindings. Keep unresolved candidate pairs available
   for transform-specific resolution; never use them as global equality constraints.
6. A known pair of distinct local landmarks stays distinct even if both carry the same colored mark.
   Transitive identity closure must not join a pair that the source explicitly distinguishes.
7. Return decisions, fixedBindings and unresolvedPairs. An unresolved visual match is a successful
   preservation of ambiguity, not an instruction to invent a preferred side.
