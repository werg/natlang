---
description: Prepare one chooseWorldActions evidence slice.
args: { goal: string; records: CharacterView[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Choose among exact legal actions in a way consistent with the character’s knowledge, goals, and relationships. Generate dialogue and creative alternatives within world and content boundaries, and avoid overriding resource or location constraints for narrative convenience. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
