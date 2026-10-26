---
description: Prepare one publishArchive evidence slice.
args: { goal: string; records: OralHistoryQuestion[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Assemble an attributed narrative and search index that preserve disagreement and access restrictions. Apply consent to quotations, identities, and derived summaries, and maintain edition history when participants correct or withdraw material. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
