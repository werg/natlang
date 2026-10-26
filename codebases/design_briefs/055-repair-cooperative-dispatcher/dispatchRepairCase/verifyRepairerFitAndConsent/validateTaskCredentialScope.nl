---
args: { diagnostic: ApprovedRepairDiagnostic, repair: CooperativeRepairRequest }
returns: CredentialTaskFit[]
---
For each diagnostic task compare its required skill with each repairer's credential scope and requested work date. Retain source evidence and expiry. Do not infer qualification from unrelated skills, identity, or prior enthusiasm.
