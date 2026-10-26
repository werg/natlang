---
description: Generate term-form candidates for an evidenced grammatical context and retain review status.
args: { dossier: TerminologyDossier; change: TerminologyChange; concepts: MultilingualConcept[] }
returns: InflectionCandidate[]
---
Use the language's supplied grammatical pattern and current reviewer evidence. Keep the lemma, grammatical context, candidate inflected forms and source versions separate. Do not silently apply a singular form in a plural context or claim a generated form is approved without a reviewer receipt.
