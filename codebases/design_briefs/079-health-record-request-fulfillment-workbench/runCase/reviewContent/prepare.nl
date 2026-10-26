---
description: Prepare one reviewContent evidence slice.
args: { goal: string; records: RecordSelection[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Assess responsiveness and identify third-party or otherwise restricted material under supplied disclosure rules. Preserve clinically relevant context where authorized, and route nuanced disclosure judgments to qualified records reviewers rather than making autonomous legal decisions. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
