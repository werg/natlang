---
description: Prepare one alignConcepts evidence slice.
args: { goal: string; records: TermOccurrence[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare conceptual boundaries across languages and teams, identifying exact, broader, narrower, and partial equivalence. Preserve concepts that do not map cleanly and avoid forcing one language’s categorization onto all others. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
