---
args: { commander: ID | null, status: string, evidence: Evidence[] }
returns: CommanderAuthorityCheck
---
Locate the currently named incident commander and the approval bound to this exact runbook step. If the commander is missing, changed, or approval cites a different scope/version, return blocked and preserve the prior authorization record without transferring it implicitly.
