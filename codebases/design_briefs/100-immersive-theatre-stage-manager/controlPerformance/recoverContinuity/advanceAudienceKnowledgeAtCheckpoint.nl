---
description: Derive audience knowledge at the new show checkpoint from actual witnessed paths.
args: { incident: ShowIncident; replay: ObservedCueReplay; dramaturgy: DramaturgyMap; checkpoint: TheatreCheckpoint; showEvidence: EvidenceItem[] }
returns: KnowledgeLedger
---
Advance each group's knowledge only for scenes its actual route receipts show it attended and for facts the scene graph says were revealed there. Preserve a participant's selected quiet route without marking them as having received the speaking-role disclosure. Do not merge audience-group knowledge when paths converge; retain the two ledgers until an evidenced common reveal occurs.
