---
args: { candidates: PolicyCandidate[], scope: string }
returns: PolicySelection
---
1. Compare source authority, effective dates, jurisdiction, and explicit supersession. Upload order never establishes precedence.
2. Compare authority, effective periods, jurisdiction or organizational scope, and explicit supersession. Return selected clauses plus unresolved precedence; never resolve authority by upload order. Host authorization still decides whether the selected policy may be used.
3. Return an explicit artifact; preserve unknowns and source coverage. This procedure proposes no production write.
