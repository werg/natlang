---
args: {"records": "DatasetCandidate[]", "use": "DatasetUse", "families": "SourceFamily[]", "session": "DatasetAuditSession"}
returns: "RecordEligibility[]"
---
For each record call distinguishPermissionScope(record, use, session.sourceEvents). Inspect source integrity, intended-task relevance, quality and known leakage as separate conditions. Quarantine unresolved provenance or consent. A license permitting analysis may prohibit distributing source documents; apply the actual intended audience and artifact form. Keep missing coverage separate from invalid records: a scarce slice does not justify admitting an unauthorized record.
