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
3. For a frame directly await a scoped NL FrameSupportClosure literal on
   {unit,primaryClauses,clauses:state.clauses}. Its numbered support policy owns these actual contributions:
   a. Name localOriginEvidence by reading the original frame's stated origin. Retain its original clause
      IDs in localOriginClauseIds; an explicit missing origin remains an unresolved premise.
   b. Follow that origin's named object/mark through actual identity statements across the supplied
      clauses. Name identityBridgeEvidence and return those original IDs in identityBridgeClauseIds.
      A reused appearance alone does not identify the anchor; competing identities stay alternatives.
   c. For each source-supported anchor follow the actual named global landmark to its original fixed
      point definition. Name fixedPointEvidence and return its IDs in fixedPointClauseIds. An identity
      bridge that only names a global landmark is not a fixed point definition. If no fixed definition
      is supplied, retain that missing dependency as unresolved rather than supplying numeric coordinates.
   d. Name orientationEvidence for the original axes and every permitted export alternative. Return
      original IDs in orientationClauseIds; a missing selector preserves all source alternatives and
      does not choose physical handedness. Separate missing evidence from exhaustive known ambiguity.
   e. Return supportingClauseIds containing the actual cited evidence needed by these contributions,
      plus exact clause-keyed quotations for each group. No case-specific numeric point or basis table
      is selected here; interpretConvention consumes these originals to judge numeric conventions.
4. Name that actual returned closure actualClosure. Require every group/support ID identifies an
   original clause, each cited ID has exact evidence, and every nonempty exactQuote belongs to it.
   Bind original source identity; cited IDs alone do not establish interpreted semantic roles. A complete closure has original origin,
   fixed-point and orientation evidence; an unnecessary identity bridge may be empty only when the
   original frame already directly names its fixed point. Missing support keeps actual unresolved reasons.
   Union all actual returned group IDs and supportingClauseIds with the original unit.supportingClauseIds.
   Bind supportedUnit by copying every other original unit field unchanged. Gather supportingClauses
   from this union and bind unit to supportedUnit. For a landmark retain the original unit/supports
   unchanged and actualClosure:null. Parent passes these actual originals, not a preferred numeric answer.
5. For a frame call interpretConvention(unit,primaryClauses,supportingClauses), directly await and
   retain its actual ConventionInterpretation. For a landmark call interpretLandmark with that actual
   unit and originals, directly await and retain its actual LandmarkItemMeaning. Only this selected
   child is invoked. A rejected call records status=failed and its actual error; all prior results survive.
6. Check each complete convention's sourceFrame equals the unit's actual nonempty frame, and
   every returned landmark frame agrees likewise. A nonnull landmark meaning.localId must equal
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
