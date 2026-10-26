---
args: { prior: HarmonizationCheckpoint|null, request: HarmonizationRequest, corpus: MeasurementCorpus, bridge: MeasureBridge, missingness: MissingnessLedger, transforms: TransformReceipt[], slice: HarmonizedDatasetSlice|null, sensitivity: SensitivityFinding[], revision: MappingRevisionEffect }
returns: HarmonizationCheckpoint
---
Preserve every measure-definition and mapping edition. Record exact row transformation receipts and keep original units and values. A change to a mapping invalidates only validated slices that consumed the changed rule; unrelated slices remain current. Preserve missingness classification and participant-overlap evidence with each slice. Sensitivity findings describe alternatives and never add observations. Identical definition, receipt and revision evidence is a no-op.

Retain sliceEditions as the actual prior slice records plus any newly verified slice, preserving each record's rule IDs and input/output hashes. Use revision.invalidatedSliceIds to move affected hashes from validatedSliceHashes to staleSliceHashes; do not discard the per-slice mapping lineage needed by the next revision.
