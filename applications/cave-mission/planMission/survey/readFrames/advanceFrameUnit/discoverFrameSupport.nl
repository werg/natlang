---
description: Discover only the selected frame unit's source-owned convention and anchor dependencies.
args:
  unit: SurveyReadingUnit
  primaryClauses: SourceClause[]
  clauses: SourceClause[]
returns: FrameSupportClosure
---
1. Preserve unit, its actual primaryClauses and all immutable original clauses. This callable owns
   evidence discovery only. Its result is FrameSupportClosure; parent owns subsequent interpretation
   after receiving this actual result. Original data objects are the complete literal input/capture.
2. In eval directly await an nl<FrameSupportClosure> literal on {unit,primaryClauses,clauses}.
   Give it the following numbered source policy. It returns clause identities, quotes, role and uncertainty;
   coordinate/basis meanings remain the later convention child's own result.
3. Name ownedDeclarations from this unit's original primary clauses. Interpret whether they declare
   the fixed reference grid itself, a local exported coordinate frame, or missing/ambiguous frame role.
   Return frameRole reference-grid, local-export or unresolved. Return ownedConventionClauseIds only
   from those actual primary IDs; another frame's export options cannot become this unit's options.
4. For a reference-grid role name ReferenceEvidence from its own axis/map declarations. Retain their
   origin/point/orientation identities and exact quotes. Reference coordinates refer to that same grid,
   independently of a robot's facing direction. Another local frame's origin and export selector do not
   define this grid. An unstated local origin is not filled by selecting an arbitrary named landmark.
5. For a local-export role name LocalOriginEvidence from its actual owned origin declaration. Follow
   that named object's original identity bridges across supplied clauses. Keep repeated appearance
   separate from same-object evidence; multiple supported anchors remain alternatives, missing identity
   remains unresolved. Return actual localOriginClauseIds and identityBridgeClauseIds with exact quotes.
6. For each supported anchor follow its actual global name to the fixed point definition. Name
   FixedPointEvidence and return its original fixedPointClauseIds. A bridge merely naming the global
   landmark is not its point definition. Carry a missing defining premise as unresolved; do not replace
   the selected frame's origin with the unrelated frame whose point definition supplies context.
7. Name OrientationEvidence only from this selected unit's owned declarations. Return those original
   orientationClauseIds. A global axis dictionary may supply context for local direction words without
   licensing its own export alternatives; a local reflection option cannot change the fixed grid's basis.
   A missing export selector preserves complete source alternatives, separately from missing evidence.
8. Return only this selected unit's actual needed dependency IDs as supportingClauseIds and their
   nonempty exact SurveyQuotes. Prior inventory support is evidence to assess, not automatic ownership.
   Each field names its own contribution; retaining an identity bridge alone cannot claim fixed-point
   coverage. Original metadata may remain contextual support without becoming another frame's owner.
9. Consume actualClosure. Require ownedConventionClauseIds subset of unit.primaryClauseIds,
   local-origin/orientation IDs subset of those owned IDs, and all other IDs in actual original clauses.
   For reference-grid role every cited group/support ID belongs to its owned primary declaration scope;
   local-export role may retain external identity/point context while owning its origin/orientation.
   Require exact evidence for every cited ID and bind original identity. A malformed ownership result
   retains its precise error; an unresolved role/dependency retains its actual source-specific premise.
10. Return the actual typed closure. Parent unions its real cited groups into child evidence, preserves
    this result in FrameUnitResult and interprets only the selected unit in a separate subsequent call.
