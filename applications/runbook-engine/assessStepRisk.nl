---
description: Assess the operational risk class of a runbook step before execution.
args:
  step: RunbookStep
returns: StepRisk
readout: decision
---
Assess the operational risk of executing step (considering step.title, step.command, and step.description):
- "safe": strictly read-only inspection, diagnostics, status checks, or low-risk reversible cache warming that cannot disrupt traffic or lose data.
- "confirm": reversible state-changing actions such as rolling restarts, draining a single stateless replica, or toggling non-destructive feature flags that merit operator awareness.
- "destructive": irreversible or high-blast-radius actions such as dropping databases or tables, wiping persistent volumes, force-deleting stateful pods, purging queues, or terminating production clusters.
