---
description: Interpret one original definition unit without losing prior typed progress.
args:
  state: DefinitionFrontier
returns: DefinitionFrontier
---
1. If state.pendingUnits is empty, return state. Select its first actual unit and retain the existing
   completed, definitions and unresolved collections. This invocation owns only that unit's contribution.
2. Bind its original ProtocolRecord by sourceId. Bind original supporting records by the unit's actual
   supportingClauseIds, retaining full text and provenance. Require the unit quote to be a nonempty exact
   substring of its original record; missing identity is a failed unit, not a default owner or role.
3. Call interpretDefinition(unit, record, supportingRecords) and directly await its actual typed return.
   A meaning:null with a source-specific reason is semantic uncertainty. A rejected invocation or type
   error is an execution failure: preserve the actual one-sentence error, this unit and all prior accepted
   rows. These distinct outcomes never authorize a replacement interpretation guessed by this parent.
4. For nonnull meaning, require a nonempty ownerId/meaning and primary-clause supporting quotation.
   Validate every returned clauseId/exactQuote against this unit's original primary/support clauses.
   Bind Witness.sourceId/path from the actual original records, never from generated semantic prose.
   Construct DefinitionClause with id=unit.clauseId and the returned ownerId/kind/meaning/witnesses.
5. If an existing definition has the same clause ID, preserve the original unit focus and check whether
   the returned contribution is the same interpretation or a genuinely additional supported role. Keep
   genuinely different roles with their original shared clause identity; do not overwrite an earlier role
   simply because two reading units share a clause. Incompatible duplicate meanings remain unresolved.
6. Append exactly one DefinitionUnitResult with original unitId/sourceId/clauseId. Accepted means actual
   supported reading and failure:null; unresolved means reading:null/failure:null and actual reason;
   failed means reading:null and actual invocation/provenance error. Add the actual reason to frontier
   unresolved with this unit identity. Preserve all earlier completed objects and accepted rows unchanged.
7. Remove only this unit from pendingUnits. Return the new frontier with the actual accepted reading
   appended when valid. Missing semantic evidence and malformed returns do not become positive coverage;
   a later reader repair can resume from original failed identities while retaining all successes.
