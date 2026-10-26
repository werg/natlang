---
args: { location: FacilityLocationResolution, permissions: VisitAccessPermission[], patterns: ComplaintTimePattern[] }
returns: ConsentedVisitPlan
---
Choose observation windows that overlap the reported time and conditions, but only within active access consent. An attempted visit without access is recorded as failed; request a new permitted window and do not claim entry.
