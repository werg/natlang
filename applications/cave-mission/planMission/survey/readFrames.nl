---
description: Consume typed per-unit NL frame and landmark meanings before choosing physical alignment.
args:
  clauses: SourceClause[]
returns: FrameSurvey
---
1. Capture the immutable original clauses. In eval construct a scoped NL inventory literal returning
   SurveyUnitInventory and directly await it on these clauses. Its task is to identify local-frame
   convention units and individual landmark units, with original primary/supporting clause IDs and
   a focus question. Several named locations in one clause yield several reading units.
2. Inventory source scope only: sourceFrame uses the original survey name consistently across its
   units. Fixed global definitions support coordinates and landmarks, not another uncertain physical
   world. Unit focus must not preselect a position, basis alternative or physical identity relation.
3. Require nonempty original primaryClauseIds, only frame/landmark kinds, existing clause IDs and
   unique generated unitIds. An invalid inventory reports its unit and expected source/kind for repair.
   Visit every
   actual unit once. Gather its original primary/supporting clauses by those checked IDs; no guessed
   source or coordinate value may replace a missing inventory contribution.
4. For a frame unit, call interpretConvention(unit, primaryClauses, supportingClauses). Consume its actual
   ConventionInterpretation: bind generated alternative IDs, original sourceFrame, origin and basis
   from the returned meanings, then attach only parent-validated original clause-keyed quotations.
5. For a landmark unit, call interpretLandmark(unit, primaryClauses, supportingClauses). Consume its
   actual nullable LandmarkItemMeaning. If established, construct LandmarkMention from the returned
   localId, frame, position and description; attach original witnesses after exact quote validation.
6. Require returned frame keys to match the original unit.sourceFrame when supplied. For every
   established alternative or mention require nonempty evidence. Match each clauseId to its
   actual supplied original clause and each nonempty exactQuote to an original witness substring;
   copy sourceId/path from that matched witness. A missing quote fails the contribution for repair.
7. Keep a local frame's origin mention and local zero position separate from the global named anchor
   and its global position. Never merge frame names or replace a local point with a global point here.
   Physical anchor identity and repeated visual marks remain questions for later reconciliation.
8. Preserve different evidence-backed meanings and contradictions. Merge exact duplicate rows only;
   semantically disputed rows remain explicit unresolved interpretation rather than a last-row override.
   A complete finite convention ambiguity is represented by alternatives, not a blocking unknown.
9. Return actual alternatives and landmarks plus inventory/child unresolved premises. All semantic
   row fields came from awaited NL children. Conditional alignment consumes those exact typed values
   and still chooses no actual handedness until independent evidence supplies that distinction.
