---
description: Interpret competing expression mechanisms and correlated prior tickets.
args:
  clauses: SourceClause[]
returns: TheoryIndex
---
1. Identify the source's competing theory declarations. For each, retain theoryId, expressed quantity,
   biological compartment, developmental stage and its complete conditional rule with exact witnesses.
2. Separate those hypotheses from established transmission conventions. The source can fix how an
   allele is inherited while leaving how it becomes an observed trait uncertain.
3. Read each source joint ticket into a complete theory/assignment tuple with its positive integer
   mass. Add fixed accession assignments explicitly included by the prior convention. Bind original
   ticketId/accession IDs; an assignment neither listed nor fixed by that convention remains unresolved.
4. Interpret the prior convention. When tickets are joint possibilities, their listed masses belong
   to entire tuples; shared genotype or mechanism labels do not make those masses independent factors.
5. Preserve all tickets before evidence. A theory's apparent agreement with an observed trait does
   not authorize this reader to remove a ticket or renormalize its mass.
6. Return the actual theories, tickets, prior convention and supporting witnesses. Record duplicate
   incompatible ticket declarations explicitly instead of merging their masses silently.
