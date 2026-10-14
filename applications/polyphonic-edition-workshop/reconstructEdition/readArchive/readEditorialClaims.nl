---
description: Interpret actual editorial operations and uncertain target attachments.
args:
  clauses: SourceClause[]
  roles: RoleIndex
returns: EditorialIndex
---
1. Inventory original correction, annotation and proposed-aid clauses. Keep each original claim ID
   and distinguish a requested score change from a description of what a player once tried.
2. For every claim call judgeAuthority with its actual clauses, role index and governing policy clauses.
   Consume authorized true/false/null and scope. An authenticated author can still leave target uncertain.
3. Interpret the actual operation: duration replacement, written-rank shift, advisory instruction or
   unresolved operation. Keep null in nonapplicable operation fields; do not turn absent pitch advice
   into permission to change a duration or vice versa.
4. Identify the source-named candidate written event locations and whether one target is exclusive.
   A complete finite attachment ambiguity preserves all supported targets and is represented in those alternatives, not as missing source inventory. Repeated performance occurrences share the
   same source-level target assignment unless the original instruction explicitly distinguishes them.
5. Bind original claim identity and exact witnesses to the actual semantic fields. Preserve unsigned
   player suggestions and tentative copying expansions with their false/unknown authority status.
6. Return EditorialIndex. Location-specific applicability is a later scoped target judgment; this
   reader neither chooses a preferred target nor applies an operation to partbook rows.
