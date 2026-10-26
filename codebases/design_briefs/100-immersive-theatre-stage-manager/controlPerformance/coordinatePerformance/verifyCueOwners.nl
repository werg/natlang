---
description: Verify cue ownership, qualification, and critical acknowledgments against current production evidence.
args: { cues: CueNode[]; resources: ResourceTimeline; showEvidence: EvidenceItem[] }
returns: CueOwnerCheck
---
For every cue, require a named responsible owner and a current qualification where the cue is safety-critical. Confirm the owner is present and assigned within the supplied scene window. Mark required acknowledgments outstanding until supported by a current receipt identifying the owner, cue, and performance run. Never interpret a generated briefing as an acknowledgment. Return all missing owners, pending acknowledgments, and resource conflicts.
