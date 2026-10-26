---
description: Generalize a public location to the approved precision without exposing restricted coordinates.
args: { collection: CurationCollection; normalized: NormalizedBatch }
returns: string[]
---
Apply the collection disclosure rule and its current version to each observation. Produce a public location reference only at the configured precision. Preserve restricted locationRef and original precision in the secured provenance record. If the rule is absent, conflicted, or cannot be applied safely, emit a privacy blocker and no public coordinates.
