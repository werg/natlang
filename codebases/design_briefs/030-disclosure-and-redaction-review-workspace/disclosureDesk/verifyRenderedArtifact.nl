---
args: { receipts: RedactionReceipt[], renders: RenderReceipt[] }
returns: ReleaseArtifact[]
---
Bind each rendered artifact to the exact sanitized output hash. Compare rendered hash, re-extracted text hash, and visual review; any residual text-layer, layout, or redaction discrepancy blocks release and requires a new sanitized artifact plus fresh render verification.
