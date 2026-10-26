---
args: {"scene": "SceneInformation", "windows": "ChannelWindow[]", "reveals": "RevealBoundary[]", "guidance": "EvidenceItem[]"}
returns: "ChannelAssignment[]"
---
Prioritize essential comprehension and the intended reveal order. If dialogue leaves no safe description gap, propose only an approved alternative channel, extended-description version or edit review supported by guidance. Do not accelerate beyond an unprovided limit, overlap essential dialogue, or reveal a later explanation to save words. Return unresolved assignments with the specific capacity conflict and approval needed.
