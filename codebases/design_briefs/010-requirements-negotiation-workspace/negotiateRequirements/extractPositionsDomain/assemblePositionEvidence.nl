---
description: Retain classified positions with complete coverage and uncertainty accounting.
args:{scope:GoalContract;dossier:NegotiationDossier;classifications:PositionClassification[]}
returns:StageResultStakeholderPosition
---
Bind each classification to its stakeholder and source evidence. Build typed StakeholderPosition records only when required fields are supported; keep missing authority or assent as unresolved. Carry all candidate IDs into the coverage manifest, preserve decision questions, and report rejected duplicates with reasons.
