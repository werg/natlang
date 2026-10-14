---
description: Assemble coherent ability units and preserve source-defined timing and capacity roles.
args:
  clauses: SourceClause[]
returns: RuleIndex
---
1. Read the original clauses as a finite source collection. Declare local RuleInventory and its nested
   AbilityUnit/RuleContextMeaning contracts. Create one scoped NL inventory literal over these clauses:
   identify units that actually prescribe an ability or standing effect, then distinguish the clauses
   defining identity, timing, capacity, initial state, submitted choices and output conventions.
2. Ask that literal to group ability primary clauses only when the source links one condition/effect
   identity. Its focus names the interpretation question, leaving kind, controller basis and consequence
   to the later ability child. Context rows instead return their actual roles and source-grounded meaning.
   An identity convention or scheduling precedence can be complete context without being an ability.
3. Directly await this RuleInventory. Validate nonempty primary/context clause ID collections against
   original clauses, unique unit/context IDs and every governingContextId against the returned contexts.
   Require each context evidence clauseId to belong to that context and cover its owned clause IDs.
   Validate every quote as a nonempty exact substring of the cited original clause witness;
   bind sourceId/path in this parent and create RuleContext from the actual roles/meaning/uncertainty.
4. Require the union of ability primaryClauseIds and context clauseIds to cover every original clause ID.
   Supporting context may govern several abilities. Missing, foreign or unclassified source IDs require
   inventory repair before iteration; names or document order do not classify semantic roles.
5. Initialize RuleReadingFrontier with the actual pending ability units, empty completed results, checked
   typed contexts, original clauses and inventory unresolved findings. Call advanceRuleUnit.iterateOn
   with measure pendingUnits.length until no original ability unit remains. Each step visits one unit.
6. Consume the returned frontier. Each completed RuleUnitResult belongs to exactly one original unit;
   a successful reading is retained even if another unit remains semantically unresolved. Project rules
   only from actual accepted nonnull readings; retain every unresolved or failed unit and its specific
   source/operation reason. A failed unit requires repair before claiming complete rule interpretation.
7. Return RuleIndex with those rules, actual typed contexts, completed unitResults and the exact union
   of context clauseIds. Retain context and unit uncertainty in unresolved. Downstream board, reference
   and timing stages consume these roles; context is not converted into synthetic executable abilities.
