---
description: Resolve candidate concepts from discriminating evidence while keeping ambiguity explicit.
args: { collection: CurationCollection; normalized: NormalizedBatch; request: CurationRequest }
returns: CandidateSet
---
For each normalized observation, compare the described traits and method against the supplied project vocabulary and current evidence. Keep multiple candidate IDs when no observed feature distinguishes them. Treat a familiar name as a candidate string, not a verified identification. A rare observation with current evidence remains in the review set; ask an expert to inspect the evidence rather than removing it as an outlier. Preserve old interpretation versions when corrections are supplied.
