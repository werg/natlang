---
args: {"eligibility": "RecordEligibility[]", "families": "SourceFamily[]", "use": "DatasetUse"}
returns: "SliceAccounting[]"
---
For each required slice select only eligible records with established slice membership. Count distinct record IDs and distinct admitted source-family IDs exactly; preserve overlap rather than summing overlapping populations. Ask whether represented mechanisms and collection channels support the intended target population. Copied examples can increase raw row count but not independentCount. Return missing mechanisms and explicit representativeness limits; matching a target count alone is never a representativeness claim.
