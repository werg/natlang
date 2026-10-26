---
description: Prepare one assembleApplications evidence slice.
args: { goal: string; records: NavigationQuestion[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Convert chosen pathways into document, form, appointment, and submission tasks. Check consistency across forms, preserve the applicant’s own statements, and request review for uncertain answers rather than fabricating details to make a form complete. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
