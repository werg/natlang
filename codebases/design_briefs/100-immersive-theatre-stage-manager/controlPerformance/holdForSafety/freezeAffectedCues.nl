---
description: Determine which not-yet-executed cues and movement actions are frozen by the safety stop.
args: { incident: ShowIncident; production: ImmersiveProduction; checkpoint: TheatreCheckpoint }
returns: FrozenCueSet
---
Use only the observed cue receipt IDs and the stop's stated scene or venue-zone scope. Separate cues with an execution receipt before the stop from cues that were merely prepared, acknowledged, or pending. Freeze all not-yet-executed actions in the affected scope, plus dependent cues whose prerequisite is now held. Do not modify historical receipts or guess which unacknowledged cue ran. Return frozen IDs, already executed IDs, and held dependencies.
