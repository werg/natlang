---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftHandoverDossier
---
Construct a HandoverDossier draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Produce a navigable asset-linked dossier and a separate outstanding-obligations register. Verify search, links, file integrity, and recipient access, then record documentary acceptance independently from any qualified technical approval or occupancy decision.
