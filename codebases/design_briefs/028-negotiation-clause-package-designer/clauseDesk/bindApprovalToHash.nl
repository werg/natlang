---
args: { request: NegotiationPackageRequest, approvals: EvidencePage[] }
returns: NegotiationGuardrail
---
Match each approver's authority, decision time, scope and signature to the exact current draft hash. An earlier hash authorizes only that earlier package; any redline change requires refreshed approval before release.
