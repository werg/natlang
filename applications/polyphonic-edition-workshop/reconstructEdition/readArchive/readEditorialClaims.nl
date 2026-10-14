---
description: Interpret actual editorial operations and uncertain target attachments.
args:
  clauses: SourceClause[]
  roles: RoleIndex
returns: EditorialIndex
---
1. From the supplied full source-clause inventory, separate governing editorial-policy clauses from
   correction, annotation and proposed-aid clauses. Keep the original claim ID and all clauses for
   each act, including an attribution clause stored in another source document. A description of what
   a player once tried is evidence about a performance, not automatically a requested score change.
2. For every actual claim call judgeAuthority with that act's clauses, the supplied role index and
   the governing policy clauses selected in step 1. If the source leaves two possible issuers, pass
   both attributions to the child; do not choose the prestigious one before the authority judgment.
   Consume authorized true/false/null and scope. An authenticated author can still leave target uncertain.
3. Interpret the actual operation: duration replacement, written-rank shift, advisory instruction or
   unresolved operation. Keep null in nonapplicable operation fields; do not turn absent pitch advice
   into permission to change a duration or vice versa.
4. Identify the source-named candidate written event locations and whether one target is exclusive.
   A complete finite attachment ambiguity preserves all supported targets and is represented in those alternatives, not as missing source inventory. Repeated performance occurrences share the
   same source-level target assignment unless the original instruction explicitly distinguishes them.
5. Resolve each returned authority-evidence clauseId against the supplied full original clauses,
   check its nonempty quotation is an exact substring, and bind its sourceId/path to the actual
   claim. Copy the child's actual reason and scope into the claim alongside its original identity.
   Preserve unsigned player suggestions and tentative copying expansions with their false/unknown
   authority status.
6. Return EditorialIndex. Location-specific applicability is a later scoped target judgment; this
   reader neither chooses a preferred target nor applies an operation to partbook rows.
