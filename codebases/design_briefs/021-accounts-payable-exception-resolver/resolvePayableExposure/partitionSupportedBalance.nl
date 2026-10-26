---
args: {"lines": "PayableLine[]", "links": "ReceiptLink[]", "versions": "LiabilityVersion[]", "ledger": "PayableBook", "documents": "EvidencePage[]"}
returns: "BalancePartition[]"
uses: [resolvePayableExposure/readInvoiceLines/sumMinorUnits]
---
For each line call explainDiscrepancy(line, links, documents). Allocate only quantitatively supported and accepted portions, preserving disputed balance and verified credits. When all components are established, use sumMinorUnits on payable, held and credited parts in one currency/scale; require their sum to equal the active invoice amount. A foreign-currency document requires an explicitly authorized rational conversion basis and rounding rule before allocation. Do not reserve a receipt twice, consume a credit twice, or use a floating-point approximation to hide a remainder.
