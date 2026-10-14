---
description: Judge one complete joint prior tuple from its original declaration and fixed source context.
args:
  unit: TicketReadingUnit
  selectedClauses: SourceClause[]
  context: TheoryContext
returns: TicketInterpretation
---
1. Bind this original ticket unit, selected original clauses and actual prior convention/theory
   context. Create a scoped NL literal returning TicketInterpretation and directly await it on
   {unit, selectedClauses, context}. The literal owns only this ticket's semantic tuple.
2. Its first numbered step identifies this ticket's stated theory and positive joint mass from its
   primary declaration. A clause can declare several tickets; read the named unit's assignment and
   mass with their own association rather than copying a neighboring ticket's fields.
3. Its second step joins the actual fixed-assignment convention to the referenced accession clauses.
   Construct the complete genotype/cytoplasm assignment tuple from ticket-specific and explicitly
   fixed values. Preserve correlation: mass weights this whole tuple, not each assignment independently.
4. Its third step distinguishes absent decisive values, conflicting declarations and established
   assignments. Missing or contradictory decisive information returns meaning=null with concrete
   unresolved premises. Observation agreement and posterior probabilities are later stages' decisions.
5. Its fourth step returns one complete TicketMeaning or null inside TicketInterpretation. A complete
   meaning carries theoryId, actual assignments, integer mass and nonempty clause-keyed exact evidence
   for both the primary declaration and every fixed-assignment premise used. Preserve other issues.
6. Return the actual awaited TicketInterpretation. The parent checks original clause/quote integrity,
   binds source identities and inserts the actual tuple into the finite accumulated inventory.
