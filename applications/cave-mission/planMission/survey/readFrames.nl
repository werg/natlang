---
description: Interpret coordinate frames and inventory landmarks before choosing any physical alignment.
args:
  clauses: SourceClause[]
returns: FrameSurvey
---
1. Select explicit frame definitions, coordinate anchors, orientation phrases and export conventions.
   Preserve the source that states whether an ambiguity is exhaustive or whether more alternatives exist.
2. For each local survey frame, call interpretConvention(frameClauses, supportingClauses). Consume
   its returned FrameAlternative[]; several alternatives are a valid interpretation of missing metadata.
3. Inventory every named landmark with its frame, stated point when available, descriptive features
   and exact witnesses. Represent a missing position as null, not the origin of an assumed frame.
4. Treat an explicitly shared physical anchor as evidence for later alignment. Do not equate every
   object with a repeated color or shape to that anchor merely because it appears in the same folder.
5. Preserve map-facing directions separately from observer-facing expressions. A robot turning
   around changes its orientation, not the global names of the east and west landings.
6. Include enough local descriptions to distinguish an actual sample niche from its painted imitation.
   A point known only in a local frame remains local until a transform is applied.
7. A finite convention ambiguity represented by complete alternatives is not a blocking unknown.
   Return alternatives, landmark mentions and genuinely unresolved interpretation questions. Do not remove
   an explicitly supported convention because its resulting mission would be harder to perform.
