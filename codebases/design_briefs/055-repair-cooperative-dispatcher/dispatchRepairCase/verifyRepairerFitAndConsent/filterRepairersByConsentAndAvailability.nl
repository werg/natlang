---
args: { diagnostic: ApprovedRepairDiagnostic, repair: CooperativeRepairRequest, credentials: CredentialTaskFit[] }
returns: RepairerMatch[]
---
For task-credential fits that remain current, require exact task consent, an available slot, and named supervision where the repairer is a trainee. Return only complete assignments; preserve the reason for each unmatched task.
