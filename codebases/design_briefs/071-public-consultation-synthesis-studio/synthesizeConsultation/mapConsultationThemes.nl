---
description: Build an argument-to-theme coverage graph preserving minority concerns and disagreement type.
args: { docket: ConsultationDocket; arguments: ScopedArgument[]; campaigns: CampaignRelationGraph }
returns: ThemeCoverage[]
---
Call `qualifyThemeRelations(docket, arguments, campaigns)` before accepting an argument-theme link. Group by shared concern and reason, not vocabulary alone. Keep minority positions and cross-cutting arguments attached to their source submissions. Distinguish factual disagreement from value conflict only when the content supports that distinction; otherwise mark mixed or unknown. Do not infer public consensus from template repetition or raw volume.
