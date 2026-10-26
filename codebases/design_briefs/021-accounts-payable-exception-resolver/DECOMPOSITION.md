# Accounts-payable exception resolver

## Versioned liability and receipt reservation ledger

This application models an invoice as an evolving liability, not one yes/no payment decision. Corrected documents first resolve into one version lineage. A bipartite line-to-order/receipt graph then supports partial allocations. Quantity reservations belong to a legal-entity ledger spanning invoices, so two cases cannot spend the same receipt. Discrepancy interpretation and the independent payment approval boundary remain separate.

A late credit or period closure takes an invalidation branch before release. Posting reconciliation is a second public operation and consumes processor/ledger receipts by exact operation identity. Its unresolved outcome never turns into a new payment attempt. Persist disputed balances, source document versions, reserved receipt quantities, consumed credits, and historical service dates. Exact minor-unit summation is a small callable arithmetic helper; line meaning, duplicate liability, acceptance and authority decisions are NatLang.

## Verification boundary

The original domain records, five semantic gates, five operational scenarios, and independent finite gate answers remain the contract. Public operations and checkpoints belong to this application. Native loading and source-call inspection do not prove learned execution quality. Effect intent is distinct from an authorized receiver receipt.
