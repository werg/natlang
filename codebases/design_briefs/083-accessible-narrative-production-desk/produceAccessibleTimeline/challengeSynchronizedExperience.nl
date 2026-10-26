---
args: {"scenes": "SceneInformation[]", "allocation": "ChannelAssignment[]", "cues": "AccessibleCue[]", "reveals": "RevealBoundary[]", "receipts": "TargetPlayerReceipt[]"}
returns: "string[]"
---
Review continuity across scene transitions, not merely isolated cue validity. Verify that every essential fact appears through a usable channel at the intended knowledge boundary without unsupported speaker identity or motive. Require actual target-platform receipts for the exact source/artifact/cue inventory, with format, timing and reading order checks passing. A locally valid caption may still fail in the target player. Return exact missing fact, conflicting cue, stale source or absent receipt IDs; do not declare delivery from a planned format name.
