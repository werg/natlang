---
description: Append authorized expert corrections and identify only the dependent analytical derivatives for invalidation.
args: { checkpoint: CurationCheckpoint; corrections: ObservationCorrection[]; interpretations: CandidateSet }
returns: string[]
---
For every correction, verify that the expert, prior interpretation ID, observation ID and new concept candidates are explicit and that the correction evidence is current. Append a new interpretation version; do not edit the old one. Identify derivative records and analyses that cite the prior interpretation. Leave unrelated observations and aggregates untouched. If authorization or evidence is missing, preserve the old interpretation as current and return a blocker.
