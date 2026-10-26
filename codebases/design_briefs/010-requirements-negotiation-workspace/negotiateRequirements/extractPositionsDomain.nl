---
description: Extract a typed, authority-linked stakeholder position result from the project record.
args:{scope:GoalContract;dossier:NegotiationDossier}
returns:StageResultStakeholderPosition
---
Bind classifications = `classifyStakeholderRequests(dossier.project, dossier.stakeholderRecords)`. For each classification retain outcome, rationale, strength, authority basis, evidence and release phase. If authority is absent or conflicting, keep strength unknown and ask the named decision owner. Bind result = `assemblePositionEvidence(scope, dossier, classifications)`. Account for every supplied stakeholder candidate as accepted, unresolved or rejected; do not promote any emphatic wording by itself.
