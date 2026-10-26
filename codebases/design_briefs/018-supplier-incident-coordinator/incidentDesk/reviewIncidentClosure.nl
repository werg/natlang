---
args: { request: SupplierIncidentRequest, containment: StageResultContainmentPlan, correction: StageResultCorrectiveAction, recurrence: IncidentGuardrail, evidence: EvidencePage[] }
returns: StageResultIncidentClosure
---
Close only when containment release criteria, implemented corrective tasks, and post-correction verification are evidenced and owned. When materially similar recurrence contradicts the prior rationale, reopen the affected causal verification and preserve original closure history; retain residual exposure and monitoring obligations.
