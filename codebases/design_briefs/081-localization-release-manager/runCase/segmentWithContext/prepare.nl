---
description: Prepare one segmentWithContext evidence slice.
args: { goal: string; records: LocalizationRelease[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Parse source content into translatable units while preserving structured markup, placeholders, plurals, and surrounding context. Distinguish identical strings used for different purposes and attach the product state or audience needed to choose the correct meaning. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
