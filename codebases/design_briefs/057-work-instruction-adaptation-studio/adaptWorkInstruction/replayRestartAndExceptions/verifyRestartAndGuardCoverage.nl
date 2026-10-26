---
args: { protocols: RestartProtocol[], guards: InspectionGuardMap[], restarts: RestartCondition[], exceptions: LocalException[] }
returns: string[]
---
Check every interruption has one matching restart protocol with all required checks, and every exception preserves required inspections. List only missing or mismatched step invariants.
