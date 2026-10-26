---
args:{scope:CloseAssertionScope;evidence:EvidencePage[];accountKey:string}
returns:CloseBreakFinding
---
Assess this account's assertion lines against the exact source editions and period in scope. Separate confirmed reconciling items, supported explanations, unexplained amounts and incomplete source coverage. Preserve currency, scale and source-line references; do not cancel an unsupported balance against an unrelated offset. Mark materiality and the specific fact needed to resolve any remaining break.
