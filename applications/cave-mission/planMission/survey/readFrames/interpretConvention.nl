---
description: Judge one local frame's complete allowed origin and basis alternatives in scoped NL.
args:
  unit: SurveyReadingUnit
  frameClauses: SourceClause[]
  supportingClauses: SourceClause[]
  closure: FrameSupportClosure
returns: ConventionInterpretation
---
1. Preserve this actual unit and its sourceFrame key, actual frame clauses and supporting global origin/orientation evidence. Capture only
   those originals. In eval create an nl<ConventionInterpretation> literal and directly
   await it on {unit, frameClauses, supportingClauses, closure}; no preselected origin, axis vector or alternative table enters its input.
2. Consume closure.frameRole and actual ownedConventionClauseIds. A reference-grid unit maps its own
   coordinates to themselves: one identity transform from its original grid/axis declarations, independent
   of the robot's facing direction. A local-export unit takes export/origin rules only from its own owned
   clauses; other frames' export metadata cannot become its alternatives by copying the unit's label.
   For the local-export role the literal joins the actual source-named origin to its supplied fixed global point definition and
   source-established identity bridge before proposing a numeric origin. Both the local origin statement
   and global/identity support remain exact evidence in every complete alternative. It interprets which statements concern exported coordinates and which concern a person's
   viewing direction. It locates the actual local frame origin in the fixed global coordinate system.
3. It returns exactly the complete conventions licensed by the source. A missing selector preserves
   those alternatives; it does not permit extra rotations, reflections, scales or arbitrary translation.
   A stated reflection changes only its named basis direction and preserves the unaffected basis.
4. Exact vector arithmetic may express the already interpreted directional meanings. Each returned
   alternative carries its sourceFrame, origin, uVector, vVector and clause-keyed exact evidence for
   both origin and basis. Generated alternative IDs are assigned by the parent, not proof of truth.
5. Unresolved frame role or missing necessary local-export origin/axis evidence remains unresolved and yields no unsupported complete
   alternative. An exhaustive supplied ambiguity itself is not unresolved interpretation work.
6. Consume and return the actual ConventionInterpretation. The parent validates nonempty exact
   evidence and binds original identity before alignment applies these returned alternatives.
