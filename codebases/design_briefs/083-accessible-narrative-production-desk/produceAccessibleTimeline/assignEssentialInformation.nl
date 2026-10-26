---
args: {"scenes": "SceneInformation[]", "reveals": "RevealBoundary[]", "windows": "ChannelWindow[]", "guidance": "EvidenceItem[]"}
returns: "ChannelAssignment[]"
---
For each essential fact choose a supported channel/window using its narrative availability and actual capacity. Call resolveDenseScene(scene, itsWindows, itsRevealBoundaries, guidance) where essential information cannot fit. Avoid duplicating dialogue in description when that hides another essential fact. Preserve facts deferred to a transcript or approved alternate channel explicitly; a deferred fact is not delivered until that channel exists.
