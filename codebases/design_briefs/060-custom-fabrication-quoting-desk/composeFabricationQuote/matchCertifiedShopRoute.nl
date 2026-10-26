---
args: { concept: ConceptSelection, equipment: ShopCapability[], certificates: CertificateRequirement[] }
returns: ShopRouteAssessment
---
Bind capability = matchRequiredShopProcesses(concept, equipment). Bind certificates = verifyRequiredCertificateScopes(concept, certificates). The route includes only active equipment and certifications whose scope covers the required process; send gaps to an identified specialist task rather than implying shop capacity.
