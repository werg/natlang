---
args: { request: HarmonizationRequest, rowVersions: EvidencePage[], receipts: TransformReceipt[] }
returns: TransformReceipt[]
---
Bind every unit conversion to dataset, row-collection version, input/output units, input and output hash, executable version, and validation receipt. If a unit changes within a dataset, split at row-version boundaries; a dataset-level mapping cannot cross that change without exact per-version receipts.
