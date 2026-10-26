---
description: Preserve immutable observer submissions while creating typed observation records and unresolved metadata findings.
args: { collection: CurationCollection; request: CurationRequest; checkpoint: CurationCheckpoint }
returns: NormalizedBatch
---
Call `preserveObserverEnvelope(request, checkpoint)` before interpreting any field. For each new submission, construct the typed `FieldObservation` from the exact observer and method data. Bind its original timestamp, declared timezone, location reference, precision and media hashes in `rawProvenance`. Do not fabricate coordinates, exact times, or method from the written description. Return unresolved metadata for every absent or conflicting field, retaining its evidence IDs.
