---
args: {"component": "ReceiptComponent", "submission": "ExpenseSubmission", "appeal": "AppealRevision"}
returns: "PurposeShare"
---
Bind facts = locatePurposeFacts(component, submission.receiptFolders). Apply only the supplied purpose-allocation rule to those facts. Keep business and personal portions explicit as exact rational shares; require their sum to one when fully determined. Do not infer personal conduct from a merchant name or unrelated history. If a companion, personal extension or bundled inclusion changes eligibility, ask the smallest factual question that resolves that component; preserve independently supported business portions.
