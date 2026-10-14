---
description: Accumulate diplomatic notation readings from actual original source units.
args:
  clauses: SourceClause[]
returns: PartbookIndex
---
1. Invoke a scoped NL NotationUnit[] inventory on original clauses. Identify every original written
   event declaration and source witness, including contradictory copied readings. Unit IDs distinguish
   reading occurrences; eventId identifies the same written location across witnesses.
2. Preserve direct notation declarations separately from shorthand expansions whose source says they
   are tentative. Include actual governing source clauses in the unit scope; authority is decided later.
3. Check unit/source/clause membership and nonempty unit IDs. Initialize NotationFrontier with the
   actual units, original clauses, empty completedUnitIds/readings and any inventory uncertainty.
4. Call advanceLocation.iterateOn with measure pending.length. Consume its actual returned frontier;
   each step reads one original unit and preserves all prior successes or unresolved unit identities.
5. Compare completedUnitIds to the actual inventory exactly once. Return actual readings, all original
   unit IDs and unresolved issues. This reader does not select the majority reading or apply corrections.
