---
description: Cluster term occurrences by evidenced concept meaning and usage rather than lexical form.
args: { dossier: TerminologyDossier; change: TerminologyChange; occurrences: TermOccurrence[] }
returns: MultilingualConcept[]
---
Call `compareSenseContexts(dossier, change, occurrences)` and preserve candidate concept clusters. Compare definitions, product context, and regional usage. Keep an expert disagreement about a concept boundary open; a frequent term is not a deciding vote. Do not declare a regional variant inconsistent when its scope is supported.
