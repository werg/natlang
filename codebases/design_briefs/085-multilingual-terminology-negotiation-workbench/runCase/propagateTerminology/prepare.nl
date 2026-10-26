---
description: Prepare one propagateTerminology evidence slice.
args: { goal: string; records: TerminologyDecision[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Identify content using the affected concept and sense, propose context-aware revisions, and preserve unrelated homographs. Revalidate grammar and meaning after replacement and version the glossary and dependent release artifacts together. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
