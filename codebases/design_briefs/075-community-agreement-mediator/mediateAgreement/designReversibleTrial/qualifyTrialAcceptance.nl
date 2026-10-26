---
description: Bind participant acceptance to one exact trial version and its exit process.
args: { dossier: MediationDossier; options: CommunityOption[]; change: MediationChange }
returns: ParticipantTermState[]
---
Require a current response from each named participant that identifies the trial terms and version. Record acceptance per participant and term. Treat silence, ambiguous assent and acceptance of an earlier version as pending. If a participant withdraws, mark withdrawn and invoke the prior agreed exit process.
