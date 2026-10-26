---
description: Create disclosure-limited public records while retaining restricted originals and versioned expert corrections.
args: { collection: CurationCollection; normalized: NormalizedBatch; interpretations: CandidateSet; links: DuplicateGraph; checkpoint: CurationCheckpoint; corrections: ObservationCorrection[] }
returns: PublicProjection
---
Call `applyDisclosureProjection(collection, normalized)` to derive only the public location precision allowed by the current collection rule. Keep restricted location references and precise provenance in the checkpoint, never in the public derivative. Call `versionExpertInterpretations(checkpoint, corrections, interpretations)` to append each correction with its source evidence and prior interpretation ID. Invalidate only derivative IDs that depend on the superseded interpretation. Keep a provisional observation distinct from a verified one; publication readiness requires the disclosure and identity review to pass.
