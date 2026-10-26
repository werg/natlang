---
args: { selections: SpanSelection[], inspections: DocumentInspection[] }
returns: CombinedInferencePathway[]
---
For each set of selected spans across documents, test whether dates, rare roles, locations, linked identifiers or unique events combine to expose a restricted fact. State the specific fragments, the inference a recipient could make, and a mitigation that preserves required meaning. Cite each input source reference; do not infer that a single document's clearance makes the assembled set safe.
