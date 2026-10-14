---
description: Accumulate complete source joint tickets through finite typed reading steps.
args:
  clauses: SourceClause[]
returns: TheoryIndex
---
1. Call readTheoryContext(clauses). Preserve the actual competing theories, joint prior convention,
   original ticket units and supporting witnesses. This reader owns the source prior inventory;
   reconstructBelief later applies evidence and normalizes the returned joint possibilities.
2. Check each unit has its original nonempty ticketId, nonempty primary clause IDs and only existing
   source clause IDs. Require unique ticket IDs in the finite frontier. Conflicting declarations for
   one original ID belong to one unit with all their source clauses and an explicit unresolved issue.
3. Initialize TicketReadingFrontier with pending equal to the actual returned ticketUnits, completed
   empty, context equal to the actual TheoryContext and clauses equal to these original clauses.
4. Invoke advanceTicket.iterateOn on that state. Its measure is pending.length; each step consumes
   exactly its first original unit and appends one typed result. The original finite unit inventory is
   fixed. Completed successes remain present when another unit has missing or conflicting evidence.
5. Require completed ticket IDs to equal the original unit IDs exactly once. Project accepted nonnull
   tickets from those actual results; retain every unresolved or failed ticket ID with its specific
   issue. A source contradiction cannot be repaired by dropping its unit from coverage.
6. Return TheoryIndex with actual context theories/priorConvention, the accumulated complete tickets,
   their exact witnesses together with context witnesses, and all context/result issues. Every returned
   ticket retains its original positive joint mass. The reader neither filters by observations nor
   substitutes independent marginal priors for the source tuples.
