---
args: { pending: PendingIncidentAction[], incident: IncidentRecoveryRequest }
returns: ActionReconciliation[]
---
For each timed-out action query only its authorized operation-status endpoint. If it executed, record its actual result and do not retry. If it did not execute, re-evaluate current preconditions before a new proposal. If state remains unknown, hold further changes and ask the commander to choose a bounded reconciliation path.
