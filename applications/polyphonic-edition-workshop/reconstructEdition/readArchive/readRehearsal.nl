---
description: Read independent rehearsal relations without assigning unobserved voice owners.
args:
  clauses: SourceClause[]
  roles: RoleIndex
returns: RehearsalIndex
---
1. Inventory actual rehearsal reports and their source roles. Separate independent observation from
   copied summaries, conjectural practice notes and normative notation conventions.
2. Interpret each observation's section, event/voice scope and temporal or sonority relation. A report
   of two simultaneous ranks can constrain their unordered sonority without saying which voice owns each.
3. Return typed section-duration, same-onset or end-sonority constraints only when the source supports
   that exact relation. Preserve the original constraint ID and exact witnesses for observed values.
4. Record copied reports in excludedSourceIds when their observations are already represented by the
   independent original. Preserve their provenance rather than treating the copy as another vote.
5. Missing event attachment or uncertain observation stays unresolved. Do not use a pleasant sounding
   candidate edition as evidence of what the witness must have heard.
6. Return actual constraints, excluded source identities and uncertainty. Ensemble checking later
   joins these relations to performed events; this reader computes no preferred timeline.
