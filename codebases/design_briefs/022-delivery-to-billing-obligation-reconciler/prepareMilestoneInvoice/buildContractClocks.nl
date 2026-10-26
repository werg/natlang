---
args: {"docket": "BillingDocket", "prior": "ObligationClock[]"}
returns: "ObligationClock[]"
---
For each agreement schedule call parseBillingConditions(schedule, docket.projects). Then call placeEffectiveAmendments(conditions, prior, docket.agreementSources). Preserve acceptance triggers, partial billing rights, exclusions and source spans as separate facts. Build intervals only from supplied effective dates and supersession authority. If editions overlap without precedence, block the affected milestone rather than selecting the latest upload.
