---
description: Interpret one original frame unit and retain its actual child return with source-bound projections.
args:
  state: FrameReadFrontier
uses: [planMission/survey/readFrames/interpretConvention, planMission/survey/readFrames/interpretLandmark]
returns: FrameReadFrontier
---
1. If state.pendingUnits is empty, return state. Select state.pendingUnits' first actual unit; keep original clauses and all
   prior completed results unchanged. This invocation owns only this unit's actual new contribution.
2. Bind original primary/supporting clauses by the unit IDs. Require exact membership. For a landmark
   verify originalLocalName is a complete case-preserved source token in its actual primary text; for
   a frame preserve originalLocalName:null. A failed boundary records this unit, not a semantic default.
3. For a frame directly await a scoped NL SurveyReadingUnit support-selection literal on
   {unit,primaryClauses,clauses:state.clauses}. Its numbered policy follows the source-named origin/basis,
   then original anchor identity links, then the actual fixed global point/orientation definitions.
   Global definitions may lie outside this frame's file and provide context rather than a selected
   physical hypothesis. Only supportingClauseIds may change; all other actual unit fields stay unchanged.
4. In the frame branch name that actual returned unit supportedUnit. Require every support ID to
   identify an original clause, union its returned supports with original supports, then bind unit
   to that actual supportedUnit. In the landmark branch retain the original unit and its supports
   unchanged; no support-selection result is needed or inferred.
   Gather supportingClauses from those IDs, preserving original source witnesses. No caller supplies
   a preselected numeric origin, basis, landmark position or physically preferred interpretation.
5. For a frame call interpretConvention(unit,primaryClauses,supportingClauses), directly await and
   retain its actual ConventionInterpretation. For a landmark call interpretLandmark with that actual
   unit and originals, directly await and retain its actual LandmarkItemMeaning. Only this selected
   child is invoked. A rejected call records status=failed and its actual error; all prior results survive.
6. Check each complete convention's sourceFrame equals the unit's actual frame when supplied, and
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
   A semantic null/empty contribution requires the actual source-specific unresolved reason, status
   unresolved and no fabricated rows. A malformed return/boundary is failed with actual error and no
   accepted projection. Keep typed raw returns when valid; an untyped rejected return is not a typed meaning.
10. Append exactly one FrameUnitResult for this original unit, including its refined actual support
    context, status, nullable child result, projected rows, failure and unresolved premises. Remove only
    this original unit from state.pendingUnits. Prior completed results remain byte-for-byte unchanged.
11. Name the typed shorter frontier next and finish this invocation immediately with its actual
    value using eval({code:"return next;",finish:true});
    the parent iterator owns the next unit call. Finishing the finite
    frontier means every unit was attempted, not that all semantic premises were resolved. Only actual
    accepted rows reach projection; failed units can be repaired without repeating successful judgments.
