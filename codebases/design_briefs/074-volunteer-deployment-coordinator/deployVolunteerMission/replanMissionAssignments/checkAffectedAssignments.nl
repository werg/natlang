---
description: Recompute only assignments affected by changed volunteer qualification, consent, availability or mode.
args: { mission: MissionDossier; change: MissionChange; team: VolunteerTeam }
returns: CapabilityCoverage[]
---
For each team assignment, compare dependencies with the changed qualification, withdrawal, mode and consent sources. Retain unaffected assignment edges. Mark only affected requirements uncovered until a currently qualified and consenting volunteer is found. Do not substitute by headcount.
