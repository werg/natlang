---
args: { revision: TaxonomyDefinitionRevision, fit: VersionedCategoryFit[], taxonomy: TaxonomyEvolutionRequest }
returns: TaxonomyRecordImpact
---
Select records whose current memberships intersect the changed boundary. Re-evaluate each under both exact baseline and candidate definition versions. Return the exact set whose labels change and keep unaffected labels and cache entries intact. When historical split membership is not recoverable, mark the bridge coarse and do not manufacture a single historical child label.
