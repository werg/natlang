---
description: Read one original notation unit and retain its actual typed contribution.
args:
  state: NotationFrontier
returns: NotationFrontier
---
1. Return state when pending is empty. Otherwise select its first actual NotationUnit and gather
   only its original clause references, retaining any governing source identity context.
2. Directly await interpretNotation(unit, selectedClauses). A null meaning or failed invocation keeps
   this original unit ID with its concrete issue; it does not erase earlier completed readings.
3. For a returned meaning validate nonempty exact clause-keyed evidence and primary-source support.
   Bind Witness sourceId/path from the actual matched original clauses, not generated text fields.
4. Construct WrittenEvent by copying actual semantic fields and adding original eventId/sourceId and
   a unique generated readingId. Retain incompatible readings as separate events for later arbitration.
5. Append this actual reading or unresolved issue and mark only this unit completed. Remove exactly
   that unit from pending and preserve all other state. Return the strictly smaller original frontier.
