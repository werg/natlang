---
description: Prepare one simulatePlayers evidence slice.
args: { goal: string; records: PuzzleArtifact[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Run exact state-space checks and simulated player interpretations, including out-of-order discovery, lost props, repeated actions, and plausible wrong answers. Identify soft locks and confusing clues, then propose targeted repairs rather than adding arbitrary hints everywhere. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
