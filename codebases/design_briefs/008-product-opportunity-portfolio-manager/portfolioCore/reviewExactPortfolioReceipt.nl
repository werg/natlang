---
args: { receipt: FeasiblePortfolioSet, frames: OpportunityPortfolio[], evidenceState: OpportunityEvidenceState, sharedWork: SharedBenefitCluster[] }
returns: PortfolioReceiptAssessment
---
Accept a solver receipt only when the supplied constraint artifact names this exact planning frame, strategy, horizon, budget currency and minor units, capacity, solver version, estimate IDs, and shared-work IDs. Confirm that the typed derived estimate set and the shared-work identifiers exactly match those artifact lists and the receipt's lists, and that the receipt input digest equals the artifact digest. Any absent artifact, incomplete estimate set, unknown estimate, changed constraint, stale solver version, or identity mismatch rejects the receipt with its precise reason. Never infer a cost or execute a solver here. On a full match, return the receipt's feasible and infeasible sets and exact cost values as the validated result.
