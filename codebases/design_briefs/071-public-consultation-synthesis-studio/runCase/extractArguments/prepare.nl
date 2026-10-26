---
description: Prepare one extractArguments evidence slice.
args: { goal: string; records: Consultation[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Read submissions in context to separate positions, supporting reasons, factual claims, and proposed actions. Preserve conditional support and mixed views within one submission, and avoid reducing nuanced responses to a single positive or negative label. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
