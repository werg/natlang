---
description: Prepare one formMechanisms evidence slice.
args: { goal: string; records: JourneyLink[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Identify plausible administrative mechanisms such as unclear instructions or failed callbacks. Seek contradictory cases and alternative explanations, distinguish correlation from causality, and define evidence needed to test the mechanism without making clinical claims. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
