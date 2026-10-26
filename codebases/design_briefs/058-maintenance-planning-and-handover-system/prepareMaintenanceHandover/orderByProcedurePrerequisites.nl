---
args: { tasks: MaintenanceTask[], procedures: MaintenanceProcedureFit[] }
returns: ID[]
---
Order only tasks whose supplied prerequisite IDs form an acyclic chain in the cited procedure. Preserve unrelated assets as separate work streams; report rather than invent a missing prerequisite.
