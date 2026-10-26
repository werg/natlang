---
args: { trace: EscapeTrace, comparison: DefectRateComparison, histories: ProcessHistory[], evidence: Evidence[] }
returns: CausalReview
---
Build competing process mechanisms from the preserved route histories, then test each against both supporting and contradicting evidence. Use method comparability only to interpret observed defect patterns. Do not rank a person as a cause because their name appears in a correlated report; return a discriminating test where the evidence cannot distinguish mechanisms.
