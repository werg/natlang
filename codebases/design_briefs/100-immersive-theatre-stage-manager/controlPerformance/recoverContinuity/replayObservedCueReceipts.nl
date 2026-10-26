---
description: Reconstruct the executed cue path from receipts rather than the planned cue sheet.
args: { production: ImmersiveProduction; incident: ShowIncident; cues: CueSequence; showEvidence: EvidenceItem[]; checkpoint: TheatreCheckpoint }
returns: ObservedCueReplay
---
Start with the exact approved checkpoint receipt IDs and append only current, run-matched execution receipts in event-time order. Keep missing acknowledgments, planned cues, canceled cues and unknown outcomes in separate lists. If two receipts disagree on a cue or timestamp, stop replay at that point and name the contradiction; do not choose the version that makes the route complete.
