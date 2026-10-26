---
args: {"hypothesis": "string", "boundaries": "WaiverBoundary[]", "history": "EvidencePage[]"}
returns: "PolicyBoundaryTrial"
---
Perform a semantic counterfactual on each independently evidenced historical case under the proposed boundary. State its old and proposed disposition and the fact responsible for any change. Include at least the relevant negative boundaries present in the supplied history. Preserve surviving safeguards and an explicit counterexample list. Missing historical facts make that trial unresolved; do not fabricate an execution receipt or claim an external simulation was run.
