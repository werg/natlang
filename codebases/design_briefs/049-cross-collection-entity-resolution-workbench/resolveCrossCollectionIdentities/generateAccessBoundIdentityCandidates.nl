---
args: { resolution: EntityResolutionRequest }
returns: IdentityPairCandidate[]
---
Generate pairs only within collections the requester is authorized to compare. Record the key, alias, or bounded semantic evidence that caused each pair to enter review. Exclude any known hard incompatibility before retrieval. Sample pairs outside the blocking keys to estimate what the block may have missed; if collection access scope is absent, return no candidate pairs and request authorization.
