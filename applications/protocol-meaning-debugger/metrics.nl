---
description: Bind a symbolic metric reading to source-defined raw outcomes and audit its display meaning.
args:
  ownerId: string
  lineage: Lineage
  question: Question
returns: MetricView
---
1. Resolve this owner's selected metric link in lineage. Find its selected DefinitionClause; keep
   the selected clause identity distinct from archived same-named definitions. Inventory source
   clauses that explicitly define raw outcome codes and what those codes mean.
2. Call interpretMetric(ownerId, metricClause, question). Preserve its complete MetricReading,
   including countedTerm, denominator, aggregation, displayLabel, witnesses and actual localDecision.
3. Construct a compact binding context from that reading, the visible outcome-code definitions and
   question.metricMeaning. In eval declare OutcomeBinding with numeratorOutcome: 'pass' | 'fail' | null,
   countedMeaning: string | null, displayLabelCompatibility: 'compatible' | 'misleading' | 'unresolved',
   reason: string, witnesses: {sourceId:string;path:string;quote:string}[].
4. Construct nl.with<OutcomeBinding> whose changing input is the symbolic reading. Its instructions
   are: join the counted term to the source-defined operational outcome code; explain the underlying
   counted proposition; compare that proposition with the displayed label's claimed meaning. Preserve
   an absent or conflicting code definition as null and unresolved rather than selecting by label.
5. Directly await the binding literal. Consume numeratorOutcome as this parent's operational choice,
   countedMeaning as the interpreted proposition, and displayLabelCompatibility as a distinct display
   diagnosis. The target question helps detect a mismatch; it does not rewrite the implementation.
6. If a required code, denominator or aggregation is unresolved, retain the source-specific reasons
   and leave the resulting plan non-executable. A supported misleading display label does not by
   itself block execution: preserve the actual code so counterfactual probes expose the difference.
7. Keep the supported exact seam limited to pooled attempt percentage. Any other source-defined
   denominator or aggregation remains unresolved for this service instead of being disguised as supported.
8. Record a new LocalDecision at metrics.nl only after the binding literal returns, with owner/clause
   input IDs, its actual code/proposition/label relation and witnesses. Carry the child's reading decision
   separately. Return reading, numeratorOutcome, countedMeaning, displayLabelCompatibility, denominator,
   aggregation, implementationId, definitionId, both localDecisions and unresolved in MetricView.
9. Later plans build their ExecutablePlan from numeratorOutcome; semantic comparison reads the
   countedMeaning and label relation alongside source clauses. Baseline complement equality cannot
   erase a polarity distinction established by these returned source-grounded contributions.
