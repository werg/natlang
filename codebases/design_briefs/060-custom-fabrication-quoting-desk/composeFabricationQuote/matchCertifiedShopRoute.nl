---
args: { concept: ConceptSelection, equipment: ShopCapability[], certificates: CertificateRequirement[] }
returns: ShopRouteAssessment
---
Map each required operation to an active shop capability and match certificate scope to the exact requirement. Missing scope blocks only the relevant operation and identifies a qualified specialist handoff; never represent an unheld certificate as available capacity.
