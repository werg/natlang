---
args: { request: HarmonizationRequest, left: MeasureDefinition, right: MeasureDefinition, evidence: EvidencePage[] }
returns: MeasureBridge
---
Compare the two measure definitions against the requested target construct. Test instrument, population, collection method, timepoint and response scale using their source evidence. Preserve original units and missing-value meanings. Return a comparable pair only when the evidence supports the same construct and scope; otherwise state which dimension differs and leave the pair unresolved or noncomparable. A matching variable name alone is not evidence.
