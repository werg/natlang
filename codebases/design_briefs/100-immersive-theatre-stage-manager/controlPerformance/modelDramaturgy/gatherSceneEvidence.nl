---
description: Gather a bounded, version-aware source slice for one scene's beat and knowledge edges.
args: { production: ImmersiveProduction; sceneId: string; showEvidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select current approved script, rehearsal, audience-path and character-knowledge evidence that names the requested scene or an explicit prerequisite. Retain conflicting and superseded versions for comparison, but only current approved sources can establish the live reveal. Record exactly which sources and scope were searched. Name missing beat, reveal, actor-knowledge, audience-group, or route evidence separately; do not summarize away the missing distinction.
