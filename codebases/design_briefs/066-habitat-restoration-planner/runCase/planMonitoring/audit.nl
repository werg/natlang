---
args: { draft: DraftMonitoringTask, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: AuditResult
---
Independently compare every populated MonitoringTask field with source versions and a plausible rival interpretation. `verdict = await qualify(draft, prepared, decisions)` and include that child result. Gate: Monitoring can inform a real decision and does not confuse changing effort with changing conditions. Explain the decisive evidence and smallest repair target; never weaken the gate.
