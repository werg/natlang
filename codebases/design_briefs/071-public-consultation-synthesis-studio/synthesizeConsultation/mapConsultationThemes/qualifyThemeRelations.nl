---
description: Check whether each proposed theme link preserves the argument's reason, scope and minority qualification.
args: { docket: ConsultationDocket; arguments: ScopedArgument[]; campaigns: CampaignRelationGraph }
returns: string[]
---
For each argument, verify an explicit relation to the theme's common concern. Keep its position and scope when reasons overlap but conclusions differ. Flag a theme that merges incompatible proposals, omits a minority concern, or implies consensus solely from repeated template language. Every argument must map to a theme or receive a named disposition.
