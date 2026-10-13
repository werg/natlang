---
description: Translate one frame convention into explicit origin and basis alternatives.
args:
  frameClauses: SourceClause[]
  supportingClauses: SourceClause[]
returns: FrameAlternative[]
---
1. Identify the local frame, its origin and the meaning of each positive basis direction. Resolve
   which statements concern exported coordinates and which concern a person's viewing direction.
2. Enumerate exactly the conventions permitted by the frame clauses. A missing selector preserves
   those alternatives; it does not authorize arbitrary rotations, scale changes or additional mirrors.
3. For each convention, return an origin Point and uVector/vVector in global coordinates. A stated
   reflection reverses only the named axis; preserve all explicitly unaffected axes and scale.
4. Use exact arithmetic for vector components only after interpreting the directional statement.
   Carry witnesses for both the basis convention and the physical origin binding.
5. If a necessary origin or axis mapping remains unsupported, do not fill it with zero by habit.
   Return only supported complete alternatives; the caller records the missing constraint as unresolved.
6. Give each alternative a unique generated identifier and the exact sourceFrame name. Its ID may
   describe the convention but must not imply that this alternative has already been selected as true.
7. Return all supported alternatives. Applying them to landmarks and rejecting contradictions are
   separate contributions owned by alignFrames and world construction.
