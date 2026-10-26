---
args: { promises: VersionedDocumentationPromise[], probes: BehaviorProbePlan[], mismatches: DocumentationDiscrepancy[], rebase: DocumentationRebaseReview, documentation: DocumentationEditionRequest }
returns: DocumentationReleaseDecision
---
Require every published claim and executable example to bind to the exact target release. Block on an unresolved authoritative mismatch, failed example, stale page base, or missing probe observation required by the release policy. Keep translation edition lag separate from whether the translated words are accurate. Return the release bundle version and exact unresolved items.
