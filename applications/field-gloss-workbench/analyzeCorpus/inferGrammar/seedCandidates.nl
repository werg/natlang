---
description: Interpret source-visible finite choice domains without selecting a winner.
args:
  source: CorpusSource
  graph: EvidenceGraph
returns: CandidateInventory
---
1. Read the actual source candidate-slot meanings, permitted values and whether their cross-product is explicitly closed. Preserve unknown domain completeness as an unresolved premise.
2. Identify structural compatibility rules declared for candidate combinations. Semantic evidence that favors a candidate belongs to later attempts, not hidden seeding elimination.
3. Enumerate each exact permitted choice combination once with a generated hypothesisId. Exact Cartesian enumeration and ID uniqueness are mechanism; each slot retains its source-defined interpreted meaning and original witness.
4. Distinguish a frozen-form candidate's declared lexical inventory from a productive-affix candidate's declared reuse rule. Do not expand a frozen inventory to accommodate an inconvenient later attestation.
5. Create pending GrammarHypothesis rows and preserve every permitted combination, including ones expected to be contradicted. No final gloss labels, survivor IDs or reference certificate are inputs.
6. Verify unique original slot/value IDs and one choice per required slot. A malformed combination has an explicit unresolved domain reason, never a silent duplicate candidate.
7. Return actual complete CandidateInventory. Its explicit finite choices determine exact coverage; later NL interpretation determines linguistic support.
