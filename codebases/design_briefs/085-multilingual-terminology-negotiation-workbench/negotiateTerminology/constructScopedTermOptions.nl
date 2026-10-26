---
description: Propose context-scoped terms and grammatical forms for each agreed concept.
args: { dossier: TerminologyDossier; change: TerminologyChange; concepts: MultilingualConcept[] }
returns: TermOptionSet
---
Call `deriveContextualInflections(dossier, change, concepts)` and bind each form to its grammatical context and review state. Offer a term for each language, concept and usage condition. Preserve required regional variants as scoped options rather than forcing a single global string. Generate inflection candidates only for supplied grammatical contexts, keep each form linked to its concept, and mark them pending qualified review when the pattern is not already approved.
