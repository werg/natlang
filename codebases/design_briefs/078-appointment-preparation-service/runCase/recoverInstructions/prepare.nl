---
description: Prepare one recoverInstructions evidence slice.
args: { goal: string; records: AppointmentPreparation[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Parse the approved instruction set into actions, timing, exceptions, and contact routes. Preserve clinical wording where transformation could change meaning, and identify ambiguity for staff resolution rather than interpreting medical instructions independently. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
