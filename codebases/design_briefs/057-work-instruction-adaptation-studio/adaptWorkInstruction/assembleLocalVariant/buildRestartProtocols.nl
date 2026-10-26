---
args: { restarts: RestartCondition[], steps: ID[], guards: InspectionGuardMap[] }
returns: RestartProtocol[]
---
For each interrupted step bind its saved state to the exact local step and guard map. Carry all isolation checks and repeat verifications into the restart protocol. If an interrupted step has no safe saved state or does not match the adapted step, mark it incomplete and block restart.
