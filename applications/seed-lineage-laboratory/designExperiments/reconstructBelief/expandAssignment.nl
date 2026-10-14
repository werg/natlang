---
description: Consume one joint ticket under actual pedigree constraints.
args:
  state: AssignmentFrontier
returns: AssignmentFrontier
---
1. If no ticket remains, return the state finished. Otherwise select one original pending ticket and
   find its complete theory/assignment tuple in state.theories.tickets.
2. Join each assigned accession with its returned genotype/cytoplasm domain and established parental
   relations. Interpret transmission using state.pedigree.inheritanceRules and allowed parent domains.
3. Judge the whole tuple jointly. A child genotype requiring an unavailable parental allele is
   impossible even when its expression theory could explain a later observation.
4. If impossible, append a WorldRejection naming that ticket and exact contradictory pedigree
   witnesses. If admissible, append a JointWorld carrying the original ticket mass and assignments.
5. If a needed parental rule is unresolved, retain the tuple with its uncertainty explicitly recorded;
   do not convert an unknown relation to a biological exclusion.
6. Remove this one original ticket ID from pendingTicketIds. Return the actual augmented state with
   finished true exactly when the original pending set is empty. Every continuing step consumes a ticket.
