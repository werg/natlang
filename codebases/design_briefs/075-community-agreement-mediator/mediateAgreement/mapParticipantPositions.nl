---
description: Map each participant's stated interest and position to exact terms while preserving disagreement.
args: { dossier: MediationDossier; change: MediationChange; history: ParticipantTermState[] }
returns: MediationPosition[]
---
Use only participant-attributed statements. Distinguish positions, underlying stated interests and concrete requests. Preserve cross-term differences; agreeing on a closing time does not imply agreement on shared-resource access. Do not infer motive or treat silence as a position.
