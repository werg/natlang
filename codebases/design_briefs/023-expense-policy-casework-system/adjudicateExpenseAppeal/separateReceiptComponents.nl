---
args: {"folders": "EvidencePage[]", "prior": "ReceiptComponent[]"}
returns: "ReceiptComponent[]"
---
For each source receipt call readReceiptArithmetic(receipt). Partition base charges, personal companion items, bundled inclusions, taxes, tips and fees without losing their original grouping. Keep the original merchant description separate from the claimant's business explanation. Retain stable component identity across corrected receipt versions and mark unreadable amount, currency or date explicitly.
