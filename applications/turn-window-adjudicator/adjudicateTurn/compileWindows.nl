---
description: Compile source timing, snapshot ownership and finite consequence capacity into a window protocol.
args:
  source: TurnSource
returns: WindowProtocol
---
1. Inventory actual RuleReadings and source.rules.contexts with timing, identity or capacity roles.
   Consume those actual context meanings/witnesses as governing constraints, retaining their uncertainty.
   A context row contributes protocol meaning without an executable rule identity. Keep replacement applicability
   separate from triggered consequence eligibility: they observe different versions of the same event.
2. For each reading call classifyTiming(rule, source.rules, source.clauses). Consume its actual phase,
   snapshot basis, before/after references, printed priority and unresolved conditions.
3. Join source before/after requirements by their interpreted rule identities. Build a TimingEdge only
   when the source establishes precedence. Independent sibling effects stay a simultaneous group, not
   a convenient total order obtained from filenames or array positions.
4. Call checkWindowDependencies(timing, precedence). Consume orderedRuleIds, independentGroups and
   cyclicGroups. A contradiction in source priority remains unresolved before effects can be committed.
5. Identify source-defined work capacity: submitted intents are finite; one-use trigger consequence
   slots belong to their exact token identities; replacement histories belong to their declared original
   part lineage or shared card token. These capacities must not be merged into one generic counter.
6. Record the source clauses requiring one resumed before-batch snapshot for all sibling conditions.
   Their simultaneousRuleIds constrain both replacement interpretation and final damage/departure logic.
7. Return timing, precedence, dependencies, simultaneousRuleIds, workCapacityRuleIds and unresolved.
   Later capture allocates the finite work universe, while window advancement consumes this protocol's
   actual phase and snapshot distinctions. This compilation does not decide any event's outcome.
