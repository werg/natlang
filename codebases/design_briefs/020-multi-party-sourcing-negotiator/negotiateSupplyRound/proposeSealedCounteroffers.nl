---
args: {"candidates": "PackageCandidate[]", "mandate": "MandateBoundary", "book": "NegotiationBook"}
returns: "SupplierEnvelope[]"
---
For each candidate needing negotiation, select one permitted concession whose exchange improves the whole package without crossing hard terms. Call redactRivalInformation(candidate, supplierId, mandate) for each affected supplier. Make every cross-supplier dependency explicit as a buyer-side condition without disclosing private offers. Reuse the existing envelope ID for the same unchanged proposal; new terms require a new operation identity and fresh approval. Return draft envelopes only, with no fabricated dispatch receipt.
