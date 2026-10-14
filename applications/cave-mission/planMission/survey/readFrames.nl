---
description: Reduce original frame units and preserve each actual returned convention or landmark contribution.
args:
  clauses: SourceClause[]
returns: FrameSurvey
---
1. Capture immutable original clauses. Directly await a scoped NL SurveyUnitInventory literal on them.
   Its numbered inventory policy identifies frame-convention units and individual landmark units,
   original primary/supporting IDs, consistent sourceFrame and each local interpretation focus.
2. For every landmark unit return originalLocalName as the exact source-local name identifying that
   item, with its original primary clause. Several named items in one clause require distinct units.
   For frame units originalLocalName is null. Names and source scope are inventory data; position,
   basis alternative, descriptive meaning and physical identity remain the later children's judgments.
3. Consume the actual inventory. Require unique original unit IDs, frame/landmark kinds, nonempty
   existing primary IDs and existing support IDs. A landmark name must be a nonempty complete lexical
   token in its primary clause text, preserving case: adjacent letters, digits or underscore belong to
   the same token. A frame name slot is null. Retain exact failed inventory identities for repair.
4. Initialize FrameReadFrontier with original clauses, actual validated pending units, completed:[]
   and inventory findings. Name this initialState; directly await
   advanceFrameUnit.iterateOn(initialState).withMeasure(next => next.pendingUnits.length)
   .until(next => next.pendingUnits.length === 0). The original-unit count decreases structurally.
   Each invocation consumes one original unit and carries
   completed actual returns forward; continuing a partial frontier does not revisit completed units.
5. Consume the actual settled frontier. Every original unit has exactly one completed result. Accepted
   results retain their actual child meaning and checked projection; semantic uncertainty retains its
   actual nullable child return/reason; execution/provenance failure retains its precise failure.
   Preserve successful earlier contributions when a later unit fails, with exact unit coverage.
6. Form alternatives and landmarks only by collecting those actual completed projections. Generated
   alternative IDs are unique across units. Merge exact duplicate semantic rows only, retaining their
   original evidence; disputed interpretations remain separate. Exhaustive convention alternatives are
   complete source ambiguity, not unfinished interpretation. This stage performs no new convention judgment.
7. Return those actual alternatives/landmarks, completed as unitResults, and carried unresolved/failure
   premises. Downstream physical alignment receives those values and original names. Its actual
   reflection choice and optional-pair consistency remain distinct later NL decisions.
