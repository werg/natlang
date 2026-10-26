---
args: { output: ApprovedAnalysisOutput, design: EffectiveDesignTimeline[], approvals: EvidencePage[] }
returns: { snapshotId: string; designVersionMatches: boolean; approvalMatches: boolean; sourceRefs: string[]; issues: string[] }
---
Match the output's snapshot and effective design version against the supplied approval record and plan timeline. Confirm the approval authorizes this analysis identity and outcome window. Return the exact matching references; missing approval, stale snapshot or conflicting design stays an issue.
