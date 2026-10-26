---
args:{prior:OnboardingCheckpoint|null;request:CustomerLaunchRequest;implementation:CustomerImplementationFile;outcomes:OnboardingOutcome[];plan:OnboardingTaskPlan|null;progress:TaskProgressLedger|null;replan:OnboardingChangeProposal|null;ownerTransfer:OnboardingGuardrail;dates:OnboardingGuardrail}
returns:OnboardingCheckpoint
---
For a first launch create a case checkpoint with the exact accepted outcome editions, task dependency plan, task evidence references, owner transfer receipts and unresolved customer acceptances. On a repeated identical source set, keep the same revision. When scope or an integration changes, preserve completed tasks with matching evidence and reopen only descendants of the changed prerequisite. Record credential revocation separately from successor assignment. Keep contractual amendments distinct from internal target dates. Advance once when evidence or task status changes; never mark a proposed task complete without its completion receipt.
