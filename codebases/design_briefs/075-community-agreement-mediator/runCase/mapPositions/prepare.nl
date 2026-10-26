---
description: Prepare one mapPositions evidence slice.
args: { goal: string; records: ExistingAgreement[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Summarize each participant’s position in terms they can review and correct. Separate factual disputes, preferences, and nonnegotiable limits, and do not infer hidden motives or psychological explanations from their communication style. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
