---
description: Separate expression hypotheses, joint-prior conventions and finite original ticket units.
args:
  clauses: SourceClause[]
returns: TheoryContext
---
1. Read the original clauses for competing expression theories. For each declared theory preserve its
   original ID, affected biological compartment, developmental stage, conditional rule and exact
   source witnesses. Keep established transmission rules distinct from uncertain expression rules.
2. Interpret the prior convention as source meaning: identify whether mass belongs to a whole joint
   tuple and which accession assignments are fixed by that convention. Retain the exact convention
   and its supporting clauses; later ticket interpretation consumes this actual context.
3. Inventory every original ticket declaration across the clauses. Create one TicketReadingUnit per
   original ticketId with primary clause IDs and supporting convention/assignment clause IDs. Several
   tickets in one clause create separate units sharing that clause. The inventory carries identities
   and evidence scope; the child ticket reader will interpret each tuple's assignment and mass.
4. For each ticket unit follow original references to fixed accession genotype/cytoplasm assignments,
   theory identity and prior conventions. Include the supporting clauses that actually establish those
   facts, even when in another source document. Missing links remain explicit issues rather than guessed
   fixed values. Preserve all declarations for a repeated ticket ID in that unit.
5. Revisit the ticket-bearing source clauses and check that every declared original ticket has one
   unit. Verify original ID/quote membership exactly. Retain conflicting theory declarations and
   unsupported convention references as unresolved; apparent fit to observations does not select a theory.
6. Return the actual theories, priorConvention, ticketUnits, witnesses and unresolved issues. The
   complete ticket tuples are owned by the subsequent per-unit reader and typed accumulator.
