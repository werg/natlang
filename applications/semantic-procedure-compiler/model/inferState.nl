---
description: Infer source-backed typed state ownership with explicit unknown and temporal evidence.
args:
  requirements: RequirementSet
  scenarios: ScenarioInput[]
returns: StateSchema
---
1. Inventory every prerequisite predicate and effect in requirements. Group only facts the source
   explicitly treats as the same state; similar wording cannot equate custody with acknowledgment.
2. For each required fact propose its key, valueType and ownerRole. Decide whether a source can leave
   it unknown and which independent source event can establish, invalidate or replace its value.
3. Inspect scenario.initial and event histories for represented facts without using scenario outcomes
   as normative rules. Add the carried establishedAt timestamp when an earlier/later join needs it.
4. Preserve actor-indexed state: changing operator cannot copy the previous operator's personal
   readiness. A shared object transfer updates holder identity, not every actor's authority to use it.
5. State identity rules from source conventions. Exact IDs and casing are plumbing; unclear referents
   remain unresolved rather than being merged by a lexical key shortcut.
6. Record invariants for monotone source knowledge and event-time effects. A later observation about
   an earlier event can change knowledge but cannot change the time at which permission was exercised.
7. Return fields, identityRules, invariants and unresolved with exact source witnesses on each field.
   Module inputs/outputs and actual scenario StateAtom updates consume this declared schema.
