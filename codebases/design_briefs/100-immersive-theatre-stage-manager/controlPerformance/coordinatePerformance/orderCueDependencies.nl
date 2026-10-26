---
description: Construct an ordered cue graph from approved route prerequisites, resource windows, and cue-sheet edges.
args: { production: ImmersiveProduction; routes: VariantSet; resources: ResourceTimeline; showEvidence: EvidenceItem[] }
returns: CueScheduleDraft
---
Use the current approved cue sheet for cue IDs, prerequisites, timing windows, owners, and safety-critical markers. Add route dependencies only when the chosen variant requires them. Produce a topological order; preserve independent cues as parallel only if their resource windows do not overlap exclusively. Detect cycles and impossible windows rather than inventing an order. Each cue remains proposed until exact host receipts arrive.
