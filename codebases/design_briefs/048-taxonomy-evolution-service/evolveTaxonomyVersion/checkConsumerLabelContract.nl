---
args: { revision: TaxonomyDefinitionRevision, taxonomy: TaxonomyEvolutionRequest }
returns: TaxonomyConsumerCompatibility[]
---
For each downstream consumer compare its accepted label cardinality and schema with the candidate rule. A single-label consumer cannot silently accept a multi-label result. Return the exact incompatible field and require a reviewed adapter or consumer change before promotion.
