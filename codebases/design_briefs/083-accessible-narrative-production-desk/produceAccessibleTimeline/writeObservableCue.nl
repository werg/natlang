---
args: {"scene": "SceneInformation", "window": "ChannelWindow", "reveals": "RevealBoundary[]", "guidance": "EvidenceItem[]"}
returns: "AccessibleCue"
---
Write concise channel-appropriate text grounded in the actual scene. Describe observed action rather than inferred motive; identify a meaningful sound or off-screen speaker only with current source support. Preserve tone while avoiding future interpretation of a visible clue. Bind every conveyed fact and speaker identity to evidence, and keep exact proposed timing within the known window. If essential meaning cannot fit, return an unresolved cue for the parent to reallocate.
