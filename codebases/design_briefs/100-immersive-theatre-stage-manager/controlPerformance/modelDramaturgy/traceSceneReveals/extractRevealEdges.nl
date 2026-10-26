---
description: Interpret approved script, cue sheet and knowledge matrix evidence as typed scene reveal edges.
args: { sceneIds: string[]; showEvidence: EvidenceItem[] }
returns: RevealEdge[]
---
Read each source item for a scene transition, prerequisite, reveal, character knowledge, audience branch, or authorized message. Construct one edge per evidenced fact transfer, binding the exact source scene, receiving scene, fact, authorized characters, audience groups, and source ID. Use the full natural-language meaning of the source, including qualifiers and negative knowledge such as “does not know.” Keep a proposed edge disputed if its source is not current/approved or another current source disagrees. If the source names no recipient, do not guess a character or group; return the edge with empty recipients so the knowledge traversal can preserve it as unproven.
