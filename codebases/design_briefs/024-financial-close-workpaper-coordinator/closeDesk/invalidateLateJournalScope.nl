---
args: { request: CloseCycleRequest, receipts: EvidencePage[], signoffs: WorkpaperSignoff[] }
returns: CloseGuardrail
---
Bind late journal identity, affected account and posting date to exact workpaper dependencies. Invalidate only conclusions and approvals consuming that account/version, calculate the supported delta, and retain prior signoff as historical evidence.
