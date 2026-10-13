---
description: Join scoped request identity, normalized time and explicit causal references.
args:
  index: EvidenceIndex
  request: InvestigationRequest
returns: Correlation
---
1. Call alignTime(index, request.asOfGlobalTime) and bindAttempts(index). These analyses own different facts:
   time normalization does not establish identity, and reused bytes do not establish temporal precedence.
2. Call deriveCausality(index, identityResult, timelineResult) using both returned analyses.
3. Collect contradictions reported by either analysis. Add a conflict if a directed execution or compensation
   edge requires an event to follow another while both known normalized times strictly reverse that relation.
4. Equal timestamps do not contradict causality. Unknown time is not evidence of concurrency or an ordering.
5. Check each duplicate group against pairDecisions. An incompatible or independent pair cannot inhabit one
   duplicate group; preserve the conflicting decision rather than silently choosing a representative.
6. Keep rejected attempts in identity.attempts for audit and exclude them from successful duplicate groups.
   They are evidence of attempted incompatible reuse, not additional successful effects.
7. Return identity, timeline, edges and conflicts. Later stages must consume these joins rather than using
   filename similarity, arrival order or global key equality as a replacement for the source identity policy.
