---
description: Prepare one extractPermissions evidence slice.
args: { goal: string; records: CollectionUse[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Read donation agreements, licenses, consent statements, and amendments to identify permissions and restrictions. Distinguish ownership of a physical object from rights to reproduce or publish it, and preserve missing or conflicting documentation as uncertainty. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
