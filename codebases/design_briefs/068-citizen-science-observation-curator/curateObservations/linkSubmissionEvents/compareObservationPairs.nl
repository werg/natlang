---
description: Compare observation pairs without conflating co-location with event identity.
args: { normalized: NormalizedBatch; checkpoint: CurationCheckpoint }
returns: ObservationComparison[]
---
Pair only submissions in the review window or with an explicit shared-event reference. Evaluate media hash equality, observer identity, time overlap at the precision actually available, spatial relation at the original precision, and contextual event description. Record which comparison dimensions are missing. Do not infer exact time or location from rounded values. Return `same-event` only when the evidence concerns the same occurrence; otherwise choose independent, related, or unknown with its basis.
