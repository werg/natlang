---
description: Consume one source ability unit and preserve every completed typed reading.
args:
  state: RuleReadingFrontier
returns: RuleReadingFrontier
---
1. If pendingUnits is empty, return state. Otherwise select its first actual AbilityUnit. Keep the
   completed results and typed contexts intact; this step owns only the selected unit's new contribution.
2. Bind unitClauses by that unit's original primaryClauseIds and governingContexts by its actual
   governingContextIds. Check exact membership. Context roles and meanings come from the inventory;
   this step does not reconstruct a fixture-specific rule grouping or select an ability consequence.
3. Call interpretAbility(unit, unitClauses, governingContexts) and directly await its actual return.
   Consume one AbilityInterpretation. If meaning is null, require a nonempty source-specific unresolved
   reason and retain status=unresolved, reading=null and failure=null. A failed invocation instead
   retains status=failed, reading=null and the actual error in failure; it is not a semantic judgment.
4. For a nonnull meaning, require each cited clauseId to belong to this unit's primary clauses or
   its linked governing contexts. Validate each nonempty exactQuote in that original clause witness and
   require support from this unit's primary clauses. Governing witnesses may also support the meaning.
   Bind sourceId/path mechanically and construct RuleReading from the actual semantic fields returned.
   A valid reading has status=accepted and failure=null. A malformed identity/quote retains status=failed
   with the exact failed check and no reading; no unsupported semantic default replaces that result.
5. Check this unit has no previous completed result. Copy its original primaryClauseIds into exactly
   one RuleUnitResult and append it, preserving all earlier results unchanged. Append the returned
   semantic unresolved findings or the actual failure reason to state.unresolved with this unit identity.
   Preserve those obligations without deriving a replacement meaning or repeating completed calls.
6. Remove only this original unit from pendingUnits and return the shorter frontier. A transport or
   malformed-result failure identifies this unit for repair; earlier reducer states retain the completed
   results. Failed attempts and semantically unresolved units remain distinct typed results; neither
   erases completed readings, and neither counts as a successfully interpreted rule.
