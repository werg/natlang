---
args: {"before": "QualifiedProposition[]", "candidates": "CandidatePage[]", "desk": "PublicationDesk"}
returns: "ReleaseVerification[]"
---
For each successfully staged candidate bind checks = rereadCandidateClaims(candidate, before, desk). Re-extract the resulting page's claims from its actual full text, including qualifications and neighboring examples. Compare protected propositions for semantic equivalence and changed neighborhoods for newly introduced contradiction. Include cross-page references that still repeat the corrected claim. Any material loss or residual decisive conflict blocks that page; return its exact counterexample for patch repair, not a self-certified success.
