---
args: { check: EstimandCheck, output: ApprovedAnalysisOutput, account: EvidencePage }
returns: { accountRef: string; analysisIdMatch: boolean; populationMatch: boolean; outcomeWindowMatch: boolean; evidenceIds: string[]; issue: string | null }
---
Verify this account is from the relevant population and outcome window and explicitly addresses the output's mechanism. Preserve privacy-scoped source identity. Return the actual link evidence; do not infer dissatisfaction, success or failure from sentiment alone.
