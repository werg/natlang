---
description: Interpret and retain one original joint-ticket result while preserving prior progress.
args:
  state: TicketReadingFrontier
returns: TicketReadingFrontier
---
1. If pending is empty, return state. Otherwise bind its first actual TicketReadingUnit and gather the
   original clauses named in its primary/supporting IDs. Preserve completed and the remaining units.
2. Directly await interpretTicket(unit, selectedClauses, state.context). Bind that actual
   TicketInterpretation before any ticket construction. An invocation failure produces a failed
   TicketReadResult carrying this original ticket ID and the concrete failure; preserve earlier results.
3. For a returned null meaning, append an unresolved TicketReadResult with the actual child's issues.
   It represents an unreadable or contradictory source tuple, not a zero-mass biological possibility.
4. For a complete returned meaning, check theoryId belongs to actual context theories, mass is a
   positive integer, assignments have unique original accession IDs, and each evidence clause/quote
   matches a supplied original clause. Require nonempty exact evidence including a primary declaration.
   An invalid returned field produces a failed result with the field and original ticket ID for repair.
5. Bind Witness sourceId/path from the matched original clause witnesses and retain the child's exact
   quote. Construct JointTicket by adding unit.ticketId to the actual returned theoryId, assignments
   and mass. Append one accepted TicketReadResult with that ticket and the child's retained issues.
6. Return the frontier with exactly this first pending unit removed and exactly one actual result
   appended. Keep context, clauses and every prior result unchanged. Semantic evidence failures remain
   visible in their results; the finite reader still completes coverage of the other source units.
