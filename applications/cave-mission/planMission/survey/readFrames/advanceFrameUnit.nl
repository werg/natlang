---
description: Interpret one original frame unit and retain its actual child return with source-bound projections.
args:
  state: FrameReadFrontier
uses: [planMission/survey/readFrames/interpretConvention, planMission/survey/readFrames/interpretLandmark]
returns: FrameReadFrontier
---
1. If state.pendingUnits is empty, return state. Select state.pendingUnits' first actual unit; keep original clauses and all
   prior completed results unchanged. This invocation owns only this unit's actual new contribution.
2. Bind original primary/supporting clauses by the unit IDs. Require exact membership and a nonempty original sourceFrame. For a landmark
   verify originalLocalName is a complete case-preserved source token in its actual primary text; for
   a frame preserve originalLocalName:null. A failed boundary records this unit, not a semantic default.
3. For a frame call discoverFrameSupport(unit,primaryClauses,state.clauses). Name its actual typed
   result actualClosure; that child owns the selected-unit reference/local role, convention ownership,
   origin/identity/fixed-point dependencies and exact source quotations. This caller owns no replacement
   support judgment. For a landmark retain actualClosure:null and its original support context.
4. In the frame branch consume actualClosure. Require every group/support ID identifies an
   original clause, each cited ID has exact evidence, and every nonempty exactQuote belongs to it.
   Bind original source identity; cited IDs alone do not establish interpreted semantic roles. A complete local-export closure has original origin,
   fixed-point and orientation evidence; reference-grid closure owns only its own primary declarations; an unnecessary identity bridge may be empty only when the
   original frame already directly names its fixed point. Missing support keeps actual unresolved reasons.
   Bind the refined support set to the union of actual returned groups and supportingClauseIds.
   Original inventory support remains available in discovery input, but enters this set only when the
   actual closure retains it for this selected unit. Bind supportedUnit by copying every other original unit field unchanged. Gather supportingClauses
   from this union and bind unit to supportedUnit. For a landmark retain the original unit/supports
   unchanged and actualClosure:null. Parent passes these actual originals, not a preferred numeric answer.
5. For a frame call interpretConvention(unit,primaryClauses,supportingClauses,actualClosure), directly await and
   retain its actual ConventionInterpretation. For a landmark call interpretLandmark with that actual
   unit and originals, directly await and retain its actual LandmarkItemMeaning. Only this selected
   child is invoked. A rejected call records status=failed and its actual error; all prior results survive.
6. Check each complete convention's sourceFrame equals the unit's actual nonempty frame, and
   every returned landmark frame agrees likewise. Require actualClosure ownedConventionClauseIds
   subset of original primary IDs; local-origin/orientation IDs belong to that selected owner scope.
   A reference-grid closure cites only its own primary convention scope; foreign local export clauses
   cannot become its supported basis. A local-export closure may cite external identity/point context,
   keeping local origin/orientation ownership in its actual primary declarations.
   For reference-grid role the returned transform maps that coordinate grid to itself: exactly one
   alternative, origin (0,0), uVector (1,0), vVector (0,1). Validate this identity-map property only after
   the NL child selected reference-grid role; do not generate a missing convention here. For local-export
   role retain only the actual source-licensed alternatives, with owned orientation evidence.
   A nonnull landmark meaning.localId must equal
   unit.originalLocalName. Name, frame and provenance are exact boundaries; physical counterpart,
   location description and meaning still come from the actual NL child.
7. For every established alternative or landmark require nonempty primary/support evidence. Validate
   each clauseId in the actual supplied originals and each nonempty exactQuote as an exact substring.
   Bind Witness.sourceId/path from the matched parent-held original witness. Missing support is a failed
   contribution. Preserve both local-origin and global/bridge evidence from actual convention meanings.
8. Copy every actual convention origin/uVector/vVector into a unique generated FrameAlternative and
   the actual landmark position/description into LandmarkMention with the checked original name/frame.
   Local zero and global named-anchor position remain separate. These are mechanical projections of
   actual returned fields; a new judgment cannot substitute for a completed child's value.
9. A supported nonempty contribution is accepted, with the actual raw child and projected rows.
   A frame contribution also retains actualClosure.unresolved; a decisive missing fixed-point, origin
   or orientation dependency prevents a fully supported frame result. Keep the actual child return
   and explicit blocker without substituting a numeric fallback or re-solving a completed convention.
   A semantic null/empty contribution requires the actual source-specific unresolved reason, status
   unresolved and no fabricated rows. A malformed return/boundary is failed with actual error and no
   accepted projection. Keep typed raw returns when valid; an untyped rejected return is not a typed meaning.
10. Append exactly one FrameUnitResult for this original unit, including its refined actual support
    context, status, actualClosure as supportClosure, nullable child result, projected rows, failure
    and unresolved premises. Remove only
    this original unit from state.pendingUnits. Prior completed results remain byte-for-byte unchanged.
11. Name the typed shorter frontier next and finish this invocation immediately with its actual
    value using eval({code:"return next;",finish:true});
    the parent iterator owns the next unit call. Finishing the finite
    frontier means every unit was attempted, not that all semantic premises were resolved. Only actual
    accepted rows reach projection; failed units can be repaired without repeating successful judgments.
