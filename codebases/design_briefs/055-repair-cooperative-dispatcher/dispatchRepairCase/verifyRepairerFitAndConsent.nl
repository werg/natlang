---
args: { diagnostic: ApprovedRepairDiagnostic, repair: CooperativeRepairRequest }
returns: RepairerMatch[]
---
Bind credentials = validateTaskCredentialScope(diagnostic, repair). Bind eligible = filterRepairersByConsentAndAvailability(diagnostic, repair, credentials). Keep each task independent: an expired qualification removes only the credential it covers, but no assignment proceeds without current scope, task consent, a free slot, and named trainee supervision.
