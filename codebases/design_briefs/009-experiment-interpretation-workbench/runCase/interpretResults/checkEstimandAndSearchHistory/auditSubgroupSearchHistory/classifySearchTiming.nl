---
args: { output: ApprovedAnalysisOutput, search: EvidencePage }
returns: { searchRef: string; subgroup: string; timing: 'prespecified'|'post-hoc'|'unclear'; approvalRef: string | null; issue: string | null }
---
Match this search record to the exact analysis, subgroup, timestamp and approved analysis plan. A subgroup selected after inspecting outcome data is post-hoc even when its hypothesis is plausible. Return the source identity and unresolved timing or approval issue; do not infer a missing search record.
