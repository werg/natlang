---
description: Reduce original frame units and preserve each actual returned convention or landmark contribution.
args:
  clauses: SourceClause[]
returns: FrameSurvey
---
1. Capture immutable original clauses. Directly await an nl<SurveyUnitInventory> literal on those actual clauses.
   Its numbered inventory policy identifies frame-convention units and individual landmark units,
   original primary/supporting IDs, consistent sourceFrame and each local interpretation focus.
   Its output is the actual typed inventory; this stage receives identity/scope data only. A clause
   declaring a frame or naming a point in its stated frame contributes a reading unit. Identity links,
   reused appearances and supporting observations remain original context unless they also declare
   such a named coordinate item. Group each frame's own grid/axis/export declarations into its one
   primary frame unit; shared anchor/identity clauses remain supporting context. All original clauses
   remain available to later evidence discovery.
   The inventory's focus names the judgment still to be made; convention alternatives, positions and
   physical counterpart meanings are owned by their subsequent typed children.
2. For every landmark unit return originalLocalName as the exact source-local name identifying that
   item, with its original primary clause. Several named items in one clause require distinct units.
   For frame units originalLocalName is null. Names and source scope are inventory data; position,
   basis alternative, descriptive meaning and physical identity remain the later children's judgments.
3. Consume the actual inventory. Require unique original unit IDs, frame/landmark kinds, nonempty
   existing primary IDs and existing support IDs. Each unit must carry a nonempty sourceFrame grounded
   in its original frame context; missing frame scope remains an explicit unresolved inventory premise. A landmark name must be a nonempty complete lexical
   token in its primary clause text, preserving case: adjacent letters, digits or underscore belong to
   the same token. A frame name slot is null. Retain each excluded inventory unit ID, original clause IDs and actual missing/failed premise in
   inventory findings for repair; exclusion does not remove its original clauses from source context.
4. Initialize FrameReadFrontier with original clauses, actual validated pending units, completed:[]
   and inventory findings. Name this initialState; directly await
   advanceFrameUnit.iterateOn(initialState).withMeasure(next => next.pendingUnits.length)
   .until(next => next.pendingUnits.length === 0). The original-unit count decreases structurally.
   Each invocation consumes one original unit and carries
   completed actual returns forward; continuing a partial frontier does not revisit completed units.
5. Consume the actual settled frontier. Every validated grounded unit placed in pendingUnits has
   exactly one completed result. Excluded inventory identities remain in their actual unresolved/failed
   inventory findings, separately from completed unitResults. Accepted
   results retain their actual child meaning and checked projection; semantic uncertainty retains its
   actual nullable child return/reason; execution/provenance failure retains its precise failure.
   Preserve successful earlier contributions when a later unit fails, with exact unit coverage.
6. Form alternatives and landmarks only by collecting those actual completed projections. Generated
   alternative IDs are unique across units. Merge exact duplicate semantic rows only, retaining their
   original evidence; disputed interpretations remain separate. Exhaustive convention alternatives are
   complete source ambiguity, not unfinished interpretation. This stage performs no new convention judgment.
7. Return those actual alternatives/landmarks, completed validated units as unitResults, and carried
   excluded inventory identities plus unresolved/failure
   premises. Downstream physical alignment receives those values and original names. Its actual
   reflection choice and optional-pair consistency remain distinct later NL decisions.
