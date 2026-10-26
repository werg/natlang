---
args: { defect: TaxonomyStructureDiagnosis, taxonomy: TaxonomyEvolutionRequest }
returns: TaxonomyDefinitionRevision
---
Draft the definition, inclusion and exclusion boundaries, examples, and parent-child relations. For a semantic rename, preserve the stable ID and avoid reclassification. For a split, merge, or genuinely new concept, create explicit new identities and map each prior identity's historical meaning. Check related steward branches for contradictory boundaries before combining them.
Bind branchConflicts = compareConcurrentStewardBoundaries(taxonomy.stewardBranches, defect). Do not merge any category branch whose inclusion examples contradict another steward's exclusion examples.
