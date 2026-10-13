---
description: Resolve numerator polarity, denominator and aggregation as separate meanings.
args:
  ownerId: string
  lineage: Lineage
  question: Question
returns: MetricView
---
1. Find the selected metric clause and any outcome coding definitions it references. A display label
   of success is not itself an instruction to count passes if the selected implementation counts failures.
2. Call interpretMetric(ownerId, metricClause, codingClauses, question). Consume the actual returned
   numerator, denominator and aggregation meanings and its localDecision contribution.
3. Keep unresolved polarity or denominator explicit. A percent can be formatted without knowing
   which proposition it answers, but such formatting does not make an executable protocol plan.
4. Preserve the implementation's interpreted behavior even when it conflicts with question.metricMeaning.
   The comparison stage identifies that difference; materialization must not silently repair it.
5. Return the child's MetricView. It includes definitionId and witnesses identifying the selected
   artifact, not an archived metric with matching filename.
6. The supported toy exact seam evaluates pooled attempts. If an owner specifies another aggregation,
   mark it unresolved/non-executable instead of disguising it as pooled percentage.
7. Do not declare success and failure rates equivalent solely because baseline numerator equals its
   complement. Counterfactual trials will test the underlying proposition independently.
