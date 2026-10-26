---
args: { request: HarmonizationRequest, bridge: MeasureBridge, missingness: MissingnessLedger, transforms: TransformReceipt[], overlap: string[], rowVersions: EvidencePage[] }
returns: HarmonizedDatasetSlice
---
Construct a slice only from comparable measures, explicit missingness semantics, exact validated transforms and version-pinned rows. Retain included and excluded record identities, rule IDs, output hashes, inclusion query and overlap uncertainty. If any required transform is missing, leave that partition out and identify it.
