---
description: Record a terminology decision only after an authorized owner resolves concept boundary and scoped term.
args: { dossier: TerminologyDossier; change: TerminologyChange; concepts: MultilingualConcept[]; options: TermOption[]; inflections: InflectionCandidate[] }
returns: TerminologyDecision
---
Require explicit terminology-owner authority, a resolved concept boundary, selected language forms and reviewed grammatical inflections. Preserve dissent and evidence references. If experts disagree about meaning or any form remains unreviewed, keep the issue open; do not resolve it by popularity or frequency.
