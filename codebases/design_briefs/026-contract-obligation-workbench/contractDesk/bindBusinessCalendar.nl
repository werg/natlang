---
args: { request: ObligationReviewRequest, calendars: EvidencePage[] }
returns: ObligationGuardrail
---
Resolve the calendar expressly incorporated by the agreement, including jurisdiction, business-day definition, holidays and timezone. Preserve the computed due date only with its starting event and calendar evidence; otherwise report the deadline as unresolved.
