---
description: Append immutable original submission envelopes and identify re-submissions by exact observation identity.
args: { request: CurationRequest; checkpoint: CurationCheckpoint }
returns: string[]
---
Copy prior checkpoint envelopes unchanged. Append a submitted envelope only once for its observation ID and source-version pair. If the same ID arrives with different text, timestamp, location, or media hashes, retain both versions as a source conflict and require curator review. Never silently substitute the newer content into an older envelope.
