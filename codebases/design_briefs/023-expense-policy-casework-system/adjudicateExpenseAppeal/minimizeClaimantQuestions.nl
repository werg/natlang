---
args: {"decisions": "ExpenseLineDecision[]", "components": "ReceiptComponent[]", "evidence": "EvidencePage[]"}
returns: "ExpenseEvidenceRequest[]"
---
For each unresolved decision identify the fact whose answer would change eligibility or amount. Eliminate questions already answered by supplied evidence and combine only those sharing the same required fact. Explain the relevant rule without requesting unrelated personal account history. Mark privacy-sensitive supporting documents for restricted upload and list which component decisions depend on each answer.
