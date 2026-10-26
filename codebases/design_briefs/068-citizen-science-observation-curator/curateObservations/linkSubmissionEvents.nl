---
description: Build an event relation graph that separates duplicate submissions from independent corroboration.
args: { collection: CurationCollection; normalized: NormalizedBatch; checkpoint: CurationCheckpoint; request: CurationRequest }
returns: DuplicateGraph
---
Call `compareObservationPairs(normalized, checkpoint)` using observer, original time, available location precision, exact media hashes, and narrative context. Classify an edge as same-event only when matching evidence supports the same observed occurrence. Group several photos from one submitter under the same event without increasing event count. Keep a second observer independent unless evidence identifies a shared event. Nearby coordinates or identical concept labels alone never justify a merge. Preserve ambiguous pairs in `unresolved` rather than forcing a relation.
