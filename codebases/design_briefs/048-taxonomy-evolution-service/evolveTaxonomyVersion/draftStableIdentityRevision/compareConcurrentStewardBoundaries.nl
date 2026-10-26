---
args: { branches: TaxonomyStewardBranch[], defect: TaxonomyStructureDiagnosis }
returns: StewardBoundaryConflict[]
---
Compare changed inclusion and exclusion examples across every related branch. Identify pairs of examples that stewards classify oppositely under the same proposed boundary. Return both branch versions and require explicit resolution before merge.
