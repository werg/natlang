---
description: Interpret a source ability's timing, effect and reference semantics in one scoped NL judgment.
args:
  unitClauses: SourceClause[]
  governingClauses: SourceClause[]
returns: RuleMeaning
---
1. Bind the proposed unit's original source clauses and its governing timing/identity clauses. Locate
   the explicit source identity of this ability or standing rule; retain any missing identity as unresolved.
2. Form a compact scope with the governing source clauses. In eval declare local RuleMeaning with
   the supplied exact shape. Create nl.with<RuleMeaning> over this scope and directly await it on the
   changing unitClauses. Its new semantic task is the role separation in the next three numbered steps.
3. Ask the literal whether this unit replaces an event before resolution, observes an event after
   resolution, establishes a standing condition, or defines timing/identity. Distinguish an announced
   quantity, a rewritten quantity and an actual resolved effect as separate possible condition inputs.
4. Ask it to bind controllerBasis from source wording: captured at announcement, live current state,
   last-known departing instance, none when no controller is referenced, or unresolved when the source
   leaves that role open. A controller change cannot itself choose which reference the rule intended.
5. Ask it to return the exact source token IDs and their capacity meaning within condition/consequence,
   along with effectClass, timing and nonempty clause-keyed exact quotations. A source-defined finite
   capacity is retained as evidence; it does not justify inventing a capacity for another rule.
6. Consume and return the actual RuleMeaning. If a required interpretation is open, keep it in
   unresolved. The parent will validate quote bytes and bind provenance before any consumer applies
   this reading; this child chooses semantic roles, not the final board or the queue order.
