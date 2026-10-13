---
description: Compare propositions and materialized choices before interpreting numeric agreement.
args:
  question: Question
  baseline: PlanState
returns: Contrast[]
---
1. Inventory contrasts for population, anchor, window and metric roles across target/left/right plans.
   Keep the selected source clause IDs and their directed provenance separate from observed percentages.
2. For each role call interpretScope(question, dimension, baseline). Consume the returned Contrast
   with its source-grounded relation and LocalDecision.
3. A role is equivalent only if the source meanings cover the same predicate and scope, not merely
   because the baseline included IDs or dates happen to coincide.
4. Record different where a supported counterfactual can alter the proposition. An equality of
   numerical receipts is a possible coincidence finding, never an override of a different relation.
5. Preserve unresolved role contrasts when selected definition or event ownership remains unknown.
   They cannot become rejected hypotheses merely because no supplied probe exercises their distinction.
6. Return all role contrasts, including equivalent ones. The later causal diagnosis needs the unchanged
   roles as controls for probes that target one distinction.
7. Use source-visible contrast IDs when declared; otherwise derive stable IDs by role and owner pair.
   Do not use the order of baseline receipts as a substitute for implementation identity.
8. The caller consumes these proposition comparisons to construct hypotheses. No extra generic
   summary should merge population and metric polarity into a single vague pipeline difference.
