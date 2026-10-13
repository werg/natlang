---
description: Judge one semantic contrast under captured question and source lineage.
args:
  question: Question
  dimension: 'population' | 'anchor' | 'window' | 'metric' | 'lineage'
  baseline: PlanState
returns: Contrast
---
1. Select role-specific target, left and right definitions, materialized choices and exact witnesses.
   Capture their semantic predicates and the target requirement that controls this contrast.
2. In eval declare local ScopeJudgment with relation: 'equivalent' | 'different' | 'unresolved',
   distinguishingPremise: string, reason: string, witnesses: {sourceId:string;path:string;quote:string}[].
   Construct nl.with<ScopeJudgment> over the changing input this role's source clauses and plan fields.
3. Its instructions are: compare what these definitions mean across admissible trial worlds; identify
   a premise that could distinguish them; separate baseline coincidence from universal equivalence;
   retain missing scope evidence as unresolved and cite both definitions.
4. Directly await the literal. Consume its relation and distinguishing premise as the Contrast meaning
   and attach a localDecision describing the actual returned role judgment.
5. Preserve ordered temporal direction and numerator polarity. Equal offsets attached to different
   event types can be different; complementary metrics equal at50percent remain different meanings.
6. Return exact dimension and clause IDs with the consumed relation, meaning, witnesses and decision.
   These fields become the hypothesis premise and later repair coverage, not decorative prose.
7. An unresolved judgment states the absent premise. Do not substitute a broad claim that all
   pipelines differ somehow when this particular role is source-equivalent.
