---
args: { draft: DraftMonitoringTask, prepared: PreparedEvidence, decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Apply the exact gate to MonitoringTask: Monitoring can inform a real decision and does not confuse changing effort with changing conditions. Pass only when all decisive field relations are supported by current authorized evidence. Fail for a concrete contradiction or violated invariant. Needs-review when decisive evidence or authority is missing or unresolved.
