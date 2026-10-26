---
args: { request: HarmonizationRequest, revisions: EvidencePage[], slices: HarmonizedDatasetSlice[] }
returns: MappingRevisionEffect
---
Compare revised mapping IDs with each derived slice's actual rule IDs. Invalidate only slices using the revised rule; preserve unrelated derivations and their input/output hashes, and record the new rule version for required recomputation.
