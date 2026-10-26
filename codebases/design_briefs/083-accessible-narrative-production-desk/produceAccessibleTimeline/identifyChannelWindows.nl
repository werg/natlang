---
args: {"scenes": "SceneInformation[]", "evidence": "EvidenceItem[]", "guidance": "EvidenceItem[]"}
returns: "ChannelWindow[]"
---
Recover actual dialogue/sound occupancy and available channel windows from the timeline. Keep caption, description and transcript constraints distinct. Use supplied rate, overlap and platform rules rather than inventing universal reading-speed limits. A missing timing/capacity rule remains an edit/review question. Preserve meaningful sounds and essential dialogue when identifying possible description windows.
