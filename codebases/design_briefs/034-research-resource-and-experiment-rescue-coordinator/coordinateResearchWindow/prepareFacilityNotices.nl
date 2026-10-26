---
args: { assignments: AssignmentSearch, readiness: ReadinessDecision[], alternatives: RescueAlternative[] }
returns: ActionProposal[]
---
Prepare role-specific notices for resource owners, operators, and investigators with the proposed time, readiness blockers, and required response. Clearly mark each as a draft for a human-authorized host to send. Never emit an execution order when readiness is blocked or a protocol deviation awaits approval.
