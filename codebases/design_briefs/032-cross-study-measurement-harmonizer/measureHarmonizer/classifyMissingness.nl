---
args: { request: HarmonizationRequest, evidence: EvidencePage[] }
returns: MissingnessLedger
---
Classify observed zero, structural absence, nonresponse, censoring, below-detection, and unknown missingness separately for each measure and study. A zero is a valid value only where collection semantics establish it; never turn missing into zero or vice versa.
