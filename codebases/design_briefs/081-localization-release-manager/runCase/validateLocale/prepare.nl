---
description: Prepare one validateLocale evidence slice.
args: { goal: string; records: LocalizedSegment[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Run exact placeholder, markup, length, plural-form, and link checks, then review semantic fidelity and naturalness independently. Inspect cross-segment consistency and context screenshots or metadata, distinguishing harmless variation from contradictory terminology or behavior. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
