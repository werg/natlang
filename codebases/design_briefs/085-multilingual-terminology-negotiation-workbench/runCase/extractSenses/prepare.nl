---
description: Prepare one extractSenses evidence slice.
args: { goal: string; records: TerminologyProject[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Collect terminology in actual use across expert documents and translations, preserving grammatical and domain context. Separate different senses of the same word and identify cases where different words denote one concept rather than assuming one-to-one lexical equivalence. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
