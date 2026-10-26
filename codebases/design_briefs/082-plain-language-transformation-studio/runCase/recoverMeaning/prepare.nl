---
description: Prepare one recoverMeaning evidence slice.
args: { goal: string; records: PlainLanguageProject[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Parse the source into claims, duties, permissions, conditions, and exceptions before rewriting. Identify terms that require explanation and preserve uncertainty or scope; do not treat dense wording as permission to remove consequential distinctions. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
