---
args: { promises: VersionedDocumentationPromise[], probes: BehaviorProbePlan[], documentation: DocumentationEditionRequest }
returns: DocumentationDiscrepancy[]
---
Compare each promise with product requirements, release decisions, independent tests, and matched probe observations. A behavior mismatch is not permission to rewrite docs to match a defect. Require authoritative intent to choose between product repair and documentation repair; otherwise keep the discrepancy uncertain and ask the named owner.
