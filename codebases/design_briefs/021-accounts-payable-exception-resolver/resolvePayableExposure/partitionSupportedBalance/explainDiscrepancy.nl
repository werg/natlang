---
args: {"line": "PayableLine", "links": "ReceiptLink[]", "documents": "EvidencePage[]"}
returns: "BalancePartition"
---
Distinguish partial delivery, unauthorized substitution, contractual freight, genuine duplicate liability and unreadable amount. Ground the classification in contract terms and actual receiving evidence. Hold only the affected amount when an undisputed portion is separately supported. If amount or currency cannot be established, return null for unestablished payable/held/credited components and ask for the source; never fabricate a zero balance. Name the evidence supporting each payable or credited portion.
