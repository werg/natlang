---
description: Propose several planting and communal-use alternatives across plots, member goals and shared resources.
args: { garden: GardenDossier; change: GardenChange; preferences: GardenPreference[]; conditions: PlotConditionMap[] }
returns: PlotAlternative[]
---
Call `verifyGardenAccessAndRules(garden, conditions)` before judging any alternative. Then call `compareMemberTradeoffs(garden, preferences, conditions)` to show which member goals each alternative serves and which it limits. Include shared paths, tool needs, water or other supplied resource constraints, and preserve creativity only within those hard limits. Do not describe a concept as an agreement.
