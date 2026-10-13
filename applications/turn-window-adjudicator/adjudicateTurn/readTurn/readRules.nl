---
description: Assemble coherent ability units and preserve source-defined timing and capacity roles.
args:
  clauses: SourceClause[]
returns: RuleIndex
---
1. Read the source roles across clauses. Group clauses describing one ability or one standing rule;
   join an ability's condition, consequence, controller basis and use-capacity clauses by explicit source
   references. A nearby paragraph or shared word alone does not establish that they define the same rule.
2. Preserve separate units for replacement, after-resolution trigger, identity convention and timing
   precedence. A statement that some token was already spent belongs to state, though it may be relevant
   evidence for later applicability. Do not turn that state into a new rule overriding the ability.
3. For each coherent unit, call interpretAbility(unitClauses, governingClauses). The governing set
   includes only source timing, identity and capacity clauses needed to interpret this unit's roles.
4. Consume the returned RuleMeaning. Bind each evidence item to its original clause by clauseId;
   require a nonempty exactQuote contained in an original clause witness. Parent code binds sourceId
   and path from that original witness. A missing or mismatching quote fails this contribution for repair.
5. Construct RuleReading from the actual semantic fields and these checked witnesses. The parent's
   exact provenance binding does not change kind, controllerBasis, timing or effectClass returned by
   the child. Preserve every unresolved semantic role as an explicit finding.
6. Keep distinct rule IDs distinct even when consequences happen to match. Merge repeated readings
   only when their source identity and scope agree; contradictory clauses remain competing evidence.
7. Mark remaining state, submission and result-contract clauses as contextClauseIds with their exact
   source IDs. Return rules, contextClauseIds and unresolved. Later stages decide applicability from
   these actual readings and their window snapshots; this inventory does not precompute the turn result.
