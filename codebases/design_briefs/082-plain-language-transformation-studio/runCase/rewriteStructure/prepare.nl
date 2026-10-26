---
description: Prepare one rewriteStructure evidence slice.
args: { goal: string; records: ReaderTask[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Reorganize information around reader tasks, use clearer sentences, and introduce explanations or examples where they help. Preserve technical terms when necessary, label examples as illustrative, and avoid creating new rules through oversimplified summaries. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
