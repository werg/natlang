---
args: { request: ReplicationPlanRequest, receipts: EvidencePage[] }
returns: DatasetPin
---
Bind dataset identifier, row version, content hash and access scope to the reference input. Compare the observed dataset against the expected hash; a mismatch remains a non-exact replication and can proceed only as an explicitly declared adaptation.
