---
args: { request: CloseCycleRequest, events: EvidencePage[] }
returns: CloseGuardrail
---
Record the actual deadline and observed completion state with source times. Preserve a missed deadline as unresolved and escalated; never rewrite event times or manufacture retrospective signoff.
