---
args: { request: GrantCycleRequest, amendments: EvidencePage[] }
returns: GrantGuardrail
---
Compare amendment signature/approval time, stated effective date, and affected expense periods. Keep retroactive permission separate from future-only authorization; absent explicit retroactivity, retain earlier spending as unresolved under the prior terms.
