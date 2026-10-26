---
description: Convert current scene requirements and production observations into typed resource-time windows.
args: { production: ImmersiveProduction; scenes: TheatreScene[]; showEvidence: EvidenceItem[] }
returns: ResourceWindow[]
---
For each qualified scene, trace required cast, venue zones, props, accessible routes and stage-control roles to current production sources. Emit a ResourceWindow with exact resourceId, resourceKind, sceneId, window label, evidenceIds and confirmed/reserved/unavailable/unverified status. Keep an operator's qualification separate from equipment availability. Where the source establishes exact endpoints and timezone, normalize startsAt and endsAt to UTC timestamps with milliseconds without changing their instants; retain the original window label and evidence. Preserve the source's exclusive/shared/unknown exclusivity and any stated capacity limitation. An absent endpoint, timezone, capacity or availability remains unresolved. Do not fabricate an exact timestamp from a symbolic scene cue.
