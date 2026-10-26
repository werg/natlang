---
args: { request: SupplierIncidentRequest, incidents: SupplierIncident[] }
returns: GoalContract
---
Bind evidence to the exact supplier, reported problem, affected orders, severity and incident owner. Keep lot identity and scope unresolved until a source-backed trace connects them; do not merge separate failures solely because the supplier is shared.
