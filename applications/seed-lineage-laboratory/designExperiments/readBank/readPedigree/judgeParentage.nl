---
description: Judge the identity and authority of one proposed biological parental relation.
args:
  relationClauses: SourceClause[]
  context: SourceClause[]
returns: ParentageDecision
---
1. Bind relationClauses as the primary evidence and select context clauses that establish the source's
   distinction between recorded mating, accession labeling and a suggested pedigree.
2. Declare a local ParentageMeaning with childId, parentId, role, established, reason and supporting
   clause-keyed quotes. Create and directly await one scoped NL literal over those bound records.
3. Its numbered judgment identifies the two physical accessions, interprets maternal/paternal wording,
   and returns true only for established biological parentage. Explicitly rejected relations return
   false; unsupported or conflicting authority returns null with the remaining alternatives.
4. Consume that actual meaning. Bind sourceId/path from the original quoted clause in the parent,
   require each nonempty quote to occur exactly, and check accession IDs against original declarations.
5. Return ParentageDecision with those bound witnesses. A failed quote or absent identity becomes an
   unresolved decision and never narrows a genotype domain.
