---
description: Prepare one assembleLocaleRelease evidence slice.
args: { goal: string; records: LocaleQualityFinding[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Build a locale bundle bound to source, glossary, context, and review versions. Identify stale translations after source changes, apply explicit fallback rules, and release only approved segments without mixing incompatible editions in one user journey. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
