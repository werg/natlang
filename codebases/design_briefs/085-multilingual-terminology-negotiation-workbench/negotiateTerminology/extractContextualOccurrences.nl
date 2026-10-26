---
description: Bind term occurrences to their language, grammatical context, product use and source version.
args: { dossier: TerminologyDossier; change: TerminologyChange }
returns: TermOccurrence[]
---
Preserve the exact observed form and its surrounding text. Record language, product context, grammatical role, region and source version. One spelling may represent more than one sense; do not merge occurrences before concept analysis. A changed source creates a new occurrence version rather than rewriting an old record.
