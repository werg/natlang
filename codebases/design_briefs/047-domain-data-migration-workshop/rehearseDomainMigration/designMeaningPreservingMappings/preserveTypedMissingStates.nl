---
args: { meanings: SourceMeaningDecision[], targetSchema: string }
returns: MissingStateMapping[]
---
Map null, unknown, not applicable, and deliberately withheld to distinct target states when the target supports them. If it does not, declare the exact collision and information loss; do not map all states to a single blank value without domain-owner approval.
