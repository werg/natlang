---
description: Judge one source-scoped resource condition or transition without selecting a mission action.
args:
  unit: SurveyReadingUnit
  primaryClauses: SourceClause[]
  supportingClauses: SourceClause[]
returns: GearItemMeaning
---
1. Preserve the original gear unit and clauses. Capture only its supporting source scope; pass the
   unit/focus and primary clauses as the changing input to a scoped NL literal returning GearItemMeaning.
2. Directly await that literal. It separates initial state, available action, state transition and
   prohibition. Empty/full carrier state, unused/rigged rope and sample possession remain separate fields.
3. Interpret the stated condition and consequence of this item. Rigging may consume a resource while
   leaving map knowledge unchanged; a sample pickup may change carrier clearance without restoring rope.
   Preserve irreversibility and only source-authorized recovery, transfer or reuse.
4. Preserve sealed-sample prohibitions and mission-object requirements. An arrival without its required
   object is not a successful mission merely because it reaches the named location.
5. Return a nullable gear meaning with source subject, condition, consequence and clause-keyed exact
   supporting quotations. Generated ruleId may identify this unit; original source names stay exact.
   Missing decisive scope remains unresolved and never becomes an invented action or free recovery.
6. Consume and return this actual result. Parent exact provenance binding precedes the typed rule's
   use by crossing/action stages; no robot state or resource changes during interpretation.
