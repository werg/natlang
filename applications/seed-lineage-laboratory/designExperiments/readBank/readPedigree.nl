---
description: Infer admissible accession assignments from established genetic relations.
args:
  clauses: SourceClause[]
returns: PedigreeReading
---
1. Inventory accession declarations and their source-listed genotype/cytoplasm alternatives. Keep
   label equivalence distinct from genetic relation; a renamed seed lot does not create a new parent.
2. Collect each asserted or proposed parental relation with its own governing clauses. Call
   judgeParentage(relationClauses, clauses) for that relation and retain the actual returned decision.
3. Bind all original accession IDs and exact witnesses before using a decided relation. A null result
   preserves the unresolved relation and does not narrow a genotype domain.
4. Join established maternal and paternal claims with the source transmission convention. Infer only
   assignments impossible under every allowed parental alternative; leave possible alternatives intact.
5. Keep nuclear transmission and maternal cytoplasm inheritance as separately witnessed constraints.
   An accession's coat observation belongs to its stated tissue owner, not automatically its nucleus.
6. Return the accession domains, actual relation decisions, relevant inheritance rules and unresolved
   items. World elimination will record which original ticket each constraint affects.
