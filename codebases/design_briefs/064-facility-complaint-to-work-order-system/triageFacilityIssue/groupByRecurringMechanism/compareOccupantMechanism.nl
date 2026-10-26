---
args: { location: FacilityLocationResolution, observation: FacilityObservation, report: OccupantReport, priorFindings: FacilityRecurrenceFinding[] }
returns: FacilityReportLink
---
Compare this report with the original symptom only after both refer to the same resolved official location. Use report.issueDescription and its evidenceRef to distinguish a compatible symptom, a distinct fault and an unresolved account. Identify compatibleIssueIds only from priorFindings with supported compatible or recurrence relationships; a shared room name or a reporter's identity is not a causal link.

Return FacilityReportLink { reportId: report.reportId, compatibleIssueIds, relation: "same-mechanism", "separate" or "unresolved", blockers: the specific location or causal evidence gaps, shareScope: report.shareScope }. Keep reporterId out of the result. Do not reveal another occupant's identity or private account to explain the linkage.
