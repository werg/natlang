---
args: { prior: PortfolioDecisionCheckpoint | null, request: PortfolioDecisionRequest, evidence: OpportunityEvidenceState, assessment: PortfolioStrategyAssessment }
returns: PortfolioDecisionCheckpoint
---
For a first plan, create a checkpoint bound to this case and the approved baseline version, retaining the complete assessment, each exact sponsor-priority scenario fork, and solver receipt identities. A repeated assessment with the same evidence-state version, baseline and receipt digests is a no-op. When inputs change, append the new assessment and fork; preserve old selections and mark only options whose estimate, shared dependency or sponsor priority changed as superseded. Never replace a sponsor-approved baseline with a proposed fork. Keep investment approvals unresolved unless an approval receipt identifies the exact selected option and baseline; solver feasibility is not investment authority. Advance once for a new assessment, fork or receipt review.
