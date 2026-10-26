---
args: {"families": "SourceFamily[]", "eligibility": "RecordEligibility[]"}
returns: "string[]"
---
Compare every admitted record's family and underlying world to all proposed split assignments. Require one declared split per complete family and include linked duplicates in the same partition. Report the exact conflicting IDs and any unresolved source link. Reassigning only one copied row cannot close the leakage. Preserve held-world membership across future revisions.
