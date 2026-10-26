---
description: Prepare one indexMaterial evidence slice.
args: { goal: string; records: DocumentaryProject[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Segment interviews and archive descriptions while preserving timecodes, speaker context, and rights. Separate a speaker’s claim from independently established fact and retain qualifiers or questions that affect the meaning of an extract. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
