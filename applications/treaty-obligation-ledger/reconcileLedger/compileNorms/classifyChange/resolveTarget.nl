---
description: Resolve clause, actor and action references before any source change is applied.
args:
  instrument: Instrument
  clause: Clause
  archive: Archive
returns: TargetResolution
---
1. Resolve explicit clause labels first, checking that their owning instrument matches the reference.
   Retain the direction of replacement: changing the toll amount leaves notice and translation untouched.
2. Interpret possessives and actor roles in the source sentence. Its own promised acts bind the
   reserving debtor; duties owed to that actor have the opposite role and remain outside the reservation.
3. Join named parties with accession identities and source action names. Do not treat a notice heading
   as a new duty or the word rescue in an unauthorized circular as a competent exemption.
4. Return exact target clause IDs, actorScope, actionScope and witnesses. Empty scope is allowed only
   for a true non-change; ambiguous referents go in unresolved with the precise alternatives.
