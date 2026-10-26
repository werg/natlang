---
args: { request: GrantCycleRequest, amendments: EvidencePage[] }
returns: GrantGuardrail
---
Compare amendment signature/approval time, stated effective date, and affected expense periods. Return blockers for each earlier service period without explicit retroactivity authority, and preserve those expense IDs so the conditions and allocation steps can withhold them. Future-only permission cannot be used to qualify earlier spending.
