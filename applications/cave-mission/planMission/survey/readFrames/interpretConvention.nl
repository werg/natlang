---
description: Judge one local frame's complete allowed origin and basis alternatives in scoped NL.
args:
  unit: SurveyReadingUnit
  frameClauses: SourceClause[]
  supportingClauses: SourceClause[]
returns: ConventionInterpretation
---
1. Preserve this actual unit and its sourceFrame key, actual frame clauses and supporting global origin/orientation evidence. Capture only
   those originals. In eval create a scoped NL literal returning ConventionInterpretation and directly
   await it on {unit, frameClauses, supportingClauses}; no preselected origin, axis vector or alternative table enters its input.
2. The literal joins the actual source-named origin to its supplied fixed global point definition and
   source-established identity bridge before proposing a numeric origin. Both the local origin statement
   and global/identity support remain exact evidence in every complete alternative. It interprets which statements concern exported coordinates and which concern a person's
   viewing direction. It locates the actual local frame origin in the fixed global coordinate system.
3. It returns exactly the complete conventions licensed by the source. A missing selector preserves
   those alternatives; it does not permit extra rotations, reflections, scales or arbitrary translation.
   A stated reflection changes only its named basis direction and preserves the unaffected basis.
4. Exact vector arithmetic may express the already interpreted directional meanings. Each returned
   alternative carries its sourceFrame, origin, uVector, vVector and clause-keyed exact evidence for
   both origin and basis. Generated alternative IDs are assigned by the parent, not proof of truth.
5. Missing necessary origin or axis evidence remains unresolved and yields no unsupported complete
   alternative. An exhaustive supplied ambiguity itself is not unresolved interpretation work.
6. Consume and return the actual ConventionInterpretation. The parent validates nonempty exact
   evidence and binds original identity before alignment applies these returned alternatives.
