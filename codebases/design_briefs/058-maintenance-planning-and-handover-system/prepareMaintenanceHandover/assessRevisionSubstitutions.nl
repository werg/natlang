---
args: { changes: PartRevisionChange[], procedures: MaintenanceProcedureFit[] }
returns: string[]
---
Accept a revision substitution only when the supplied procedure or compatibility record explicitly covers both part IDs and revisions for this asset configuration. Otherwise identify the task and keep the offered part uninstalled.
